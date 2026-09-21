import asyncio
import io
import json

import pandas as pd

from app.agent_tools import (
    EvidenceIndex,
    PerformSqlToDocumentRecordsTool,
    PerformSqlToDocumentTool,
    RunContext,
    SearchDocumentTool,
    ValidateChunkAsStructuredDataTool,
)
from app.database import CHUNK_TABLE_NAME, DOCUMENT_METADATA_TABLE_NAME
from app.models.chunk import RetrievedChunk
from app.retrievers.base import Retriever
from app.services.rag_agent import execute_tool
from app.store_file.local import LocalFileStorage
from app.store_sql.local import LocalSqlStorage

from fakes import FakeEmbedder, FakeLLM, FakeVectorStorage, build_rag_service

SALES_CSV = b"region,amount\nnorth,10\nsouth,25\n"
BIG_CSV = b"n\n1\n2\n3\n4\n5\n6\n"
EMBEDDED_SALES_CSV = b"region,amount\nwest,5\neast,7\n"
EMBEDDED_UNITS_CSV = b"region,units\nwest,2\neast,3\n"


def make_workbook_bytes() -> bytes:
    """A real two-sheet xlsx. Sheet names carry a space so the tests exercise
    SQLite identifier quoting (the in-memory tables are named by sheet)."""
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer) as writer:
        pd.DataFrame({"region": ["north", "south"], "amount": [100, 200]}).to_excel(
            writer, sheet_name="Sheet A", index=False
        )
        pd.DataFrame({"region": ["north", "south"], "country": ["N", "S"]}).to_excel(
            writer, sheet_name="Sheet B", index=False
        )
    return buffer.getvalue()


class FixedRetriever(Retriever):
    """Returns fixed chunks, applying the where condition like the real
    retrievers do (at the index)."""

    def __init__(self, chunks: list[RetrievedChunk]) -> None:
        self._chunks = chunks
        self.queries: list[str] = []
        self.wheres: list = []

    async def retrieve(self, user_query, where=None) -> list[RetrievedChunk]:
        self.queries.append(user_query)
        self.wheres.append(where)
        if where is None:
            return self._chunks
        return [
            chunk
            for chunk in self._chunks
            if all(getattr(chunk, column) == value for column, value in where.items())
        ]


def make_chunk(
    chunk_id: str, plugin: str, text: str, chat_id: str | None = None, **metadata
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        source_id="s1",
        origin_source_id="s1",
        plugin=plugin,
        text=text,
        score=0.9,
        metadata=metadata,
        chat_id=chat_id,
    )


def build_tools(tmp_path, *, retriever_chunks=None, chat_id=None):
    """Real local storages plus the RAG tools.

    The corpus (all in chat-a): a text document, a directly ingested csv
    table, a big csv table, a csv source that stores no table, a real
    two-sheet workbook, and a pdf with two embedded tables stored as csvs
    sharing the pdf as their origin. `chat_id` scopes the tools to one chat.
    """
    retriever = FixedRetriever(retriever_chunks or [])
    sql_storage = LocalSqlStorage(storage_dir=tmp_path / "sql")
    file_storage = LocalFileStorage(storage_dir=tmp_path / "file")

    def table_metadata(name, schema, row_count):
        return {
            "table_name": name,
            "schema": schema,
            "row_count": row_count,
            "column_count": len(schema),
        }

    SALES_SCHEMA = [
        {"name": "region", "type": "TEXT"},
        {"name": "amount", "type": "INTEGER"},
    ]
    COUNTRY_SCHEMA = [
        {"name": "region", "type": "TEXT"},
        {"name": "country", "type": "TEXT"},
    ]

    async def setup():
        await sql_storage.create_tables()

        async def store_document(
            source_id, filename, content_type, file_bytes, is_origin=True
        ):
            path = await file_storage.upload(
                file=io.BytesIO(file_bytes),
                file_filename=filename,
                file_content_type=content_type,
            )
            await sql_storage.upsert(
                DOCUMENT_METADATA_TABLE_NAME,
                [
                    {
                        "source_id": source_id,
                        "file_path": path,
                        "file_content_type": content_type,
                        "file_filename": filename,
                        "file_orig_filename": filename,
                        "is_origin": is_origin,
                    }
                ],
            )

        async def store_chunk(
            chunk_id,
            source_id,
            plugin,
            text,
            metadata,
            origin_source_id=None,
            parent_source_id=None,
        ):
            await sql_storage.upsert(
                CHUNK_TABLE_NAME,
                [
                    {
                        "chunk_id": chunk_id,
                        "source_id": source_id,
                        "parent_source_id": parent_source_id,
                        "origin_source_id": origin_source_id or source_id,
                        "plugin": plugin,
                        "text": text,
                        "metadata": metadata,
                        "chat_id": "chat-a",
                    }
                ],
            )

        # A plain text document.
        await store_document(
            "doc-src", "report.txt", "text/plain", b"The quarterly report body."
        )
        await store_chunk(
            "text-chunk", "doc-src", "text", "The quarterly report body.",
            {"source_page_number": 1},
        )

        # A directly ingested csv table.
        await store_document("tbl-src", "sales.csv", "text/csv", SALES_CSV)
        await store_chunk(
            "sales-chunk", "tbl-src", "table", "Table: sales ...",
            table_metadata("sales", SALES_SCHEMA, 2),
        )

        # A big csv table (more rows than the result cap).
        await store_document("big-src", "big.csv", "text/csv", BIG_CSV)
        await store_chunk(
            "big-chunk", "big-src", "table", "Table: big ...",
            table_metadata("big", [{"name": "n", "type": "INTEGER"}], 6),
        )

        # A csv source that stores no table chunk.
        await store_document("empty-src", "empty.csv", "text/csv", b"n\n1\n")

        # A real two-sheet workbook (one source, several tables).
        await store_document(
            "wb-src", "book.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            make_workbook_bytes(),
        )
        await store_chunk(
            "sheet-a-chunk", "wb-src", "table", "Table: Sheet A ...",
            table_metadata("Sheet A", SALES_SCHEMA, 2),
        )
        await store_chunk(
            "sheet-b-chunk", "wb-src", "table", "Table: Sheet B ...",
            table_metadata("Sheet B", COUNTRY_SCHEMA, 2),
        )

        # A pdf (not a table) with two embedded tables stored as csvs sharing
        # the pdf origin.
        await store_document(
            "pdf-src", "report.pdf", "application/pdf", b"fake-pdf-bytes"
        )
        await store_chunk(
            "pdf-text-chunk", "pdf-src", "text", "The report body.",
            {"source_page_number": 2},
        )
        await store_document(
            "emb-1", "report_table_1.csv", "text/csv", EMBEDDED_SALES_CSV,
            is_origin=False,
        )
        await store_chunk(
            "emb-1-chunk", "emb-1", "table", "Table: report_table_1 ...",
            table_metadata("report_table_1", SALES_SCHEMA, 2),
            origin_source_id="pdf-src", parent_source_id="pdf-src",
        )
        await store_document(
            "emb-2", "report_table_2.csv", "text/csv", EMBEDDED_UNITS_CSV,
            is_origin=False,
        )
        await store_chunk(
            "emb-2-chunk", "emb-2", "table", "Table: report_table_2 ...",
            table_metadata(
                "report_table_2",
                [
                    {"name": "region", "type": "TEXT"},
                    {"name": "units", "type": "INTEGER"},
                ],
                2,
            ),
            origin_source_id="pdf-src", parent_source_id="pdf-src",
        )

        await sql_storage.close()

    asyncio.run(setup())

    rag_service = build_rag_service(
        FakeLLM(),
        retriever,
        sql_storage=sql_storage,
        file_storage=file_storage,
    )
    tools = [
        SearchDocumentTool(),
        ValidateChunkAsStructuredDataTool(),
        PerformSqlToDocumentTool(),
        PerformSqlToDocumentRecordsTool(),
    ]
    context = RunContext(chat_id=chat_id, evidence=EvidenceIndex())
    executors = {
        tool.name: tool.create_executor(rag_service, context) for tool in tools
    }
    return executors, retriever


def execute(executors, name: str, **arguments) -> str:
    return asyncio.run(execute_tool(executors, name, arguments))


# --- search_documents ---


def test_search_documents_returns_chunk_ids_and_text(tmp_path):
    tools, retriever = build_tools(
        tmp_path,
        retriever_chunks=[
            make_chunk("c1", "text", "first evidence"),
            make_chunk("c2", "table", "second evidence"),
        ],
    )

    output = execute(tools, "search_documents", query="evidence?")

    assert retriever.queries == ["evidence?"]
    # Each chunk is shown with its citation number and its ids (the model
    # passes source_id to the other table tools), then its text.
    assert (
        "[1] source_id=s1 chunk_id=c1 origin_source_id=s1\nfirst evidence" in output
    )
    assert (
        "[2] source_id=s1 chunk_id=c2 origin_source_id=s1\nsecond evidence" in output
    )


def test_search_documents_reports_no_evidence(tmp_path):
    tools, _ = build_tools(tmp_path)

    assert execute(tools, "search_documents", query="anything") == "No evidence found."


# --- validate_chunk_as_structured_data ---


def test_validate_csv_returns_one_schema(tmp_path):
    tools, _ = build_tools(tmp_path)

    output = json.loads(
        execute(tools, "validate_chunk_as_structured_data", source_id="tbl-src")
    )

    assert len(output) == 1
    assert output[0]["table_name"] == "sales"
    assert output[0]["schema"] == [
        {"name": "region", "type": "TEXT"},
        {"name": "amount", "type": "INTEGER"},
    ]
    assert output[0]["row_count"] == 2


def test_validate_workbook_returns_one_schema_per_sheet(tmp_path):
    tools, _ = build_tools(tmp_path)

    output = json.loads(
        execute(tools, "validate_chunk_as_structured_data", source_id="wb-src")
    )

    assert {entry["table_name"] for entry in output} == {"Sheet A", "Sheet B"}
    assert len(output) == 2


def test_validate_rejects_a_non_table_source(tmp_path):
    tools, _ = build_tools(tmp_path)

    output = execute(tools, "validate_chunk_as_structured_data", source_id="doc-src")

    assert output.startswith("Error in tool 'validate_chunk_as_structured_data'")
    assert "not a table" in output


def test_validate_rejects_a_source_that_stores_no_table(tmp_path):
    tools, _ = build_tools(tmp_path)

    output = execute(tools, "validate_chunk_as_structured_data", source_id="empty-src")

    assert output.startswith("Error in tool 'validate_chunk_as_structured_data'")
    assert "stores no table" in output


def test_validate_rejects_an_unknown_source(tmp_path):
    tools, _ = build_tools(tmp_path)

    output = execute(tools, "validate_chunk_as_structured_data", source_id="nope")

    assert output.startswith("Error in tool 'validate_chunk_as_structured_data'")
    assert "Unknown source" in output


# --- perform_sql_to_document ---


def test_perform_sql_computes_on_a_csv(tmp_path):
    tools, _ = build_tools(tmp_path)

    output = execute(
        tools,
        "perform_sql_to_document",
        source_id="tbl-src",
        sql_query="SELECT SUM(amount) AS total FROM sales",
    )

    assert "Tables: sales (#[1])" in output
    assert "35" in output


def test_perform_sql_joins_a_workbooks_sheets(tmp_path):
    tools, _ = build_tools(tmp_path)

    output = execute(
        tools,
        "perform_sql_to_document",
        source_id="wb-src",
        sql_query=(
            'SELECT "Sheet B".country, SUM("Sheet A".amount) AS total '
            'FROM "Sheet A" JOIN "Sheet B" '
            'ON "Sheet A".region = "Sheet B".region '
            'GROUP BY "Sheet B".country ORDER BY "Sheet B".country'
        ),
    )

    # Both sheets are loaded, and the join pairs Sheet A's amounts with
    # Sheet B's country.
    assert "N,100" in output
    assert "S,200" in output


def test_perform_sql_returns_all_rows_without_a_limit(tmp_path):
    tools, _ = build_tools(tmp_path)

    output = execute(
        tools,
        "perform_sql_to_document",
        source_id="big-src",
        sql_query="SELECT n FROM big",
    )

    # No row cap: without a LIMIT the query returns every row, and there is
    # no truncation note.
    lines = output.split("\n")
    assert lines[-6:] == ["1", "2", "3", "4", "5", "6"]
    assert "showing the first" not in output


def test_perform_sql_rejects_a_non_table_source(tmp_path):
    tools, _ = build_tools(tmp_path)

    output = execute(
        tools,
        "perform_sql_to_document",
        source_id="doc-src",
        sql_query="SELECT 1",
    )

    assert output.startswith("Error in tool 'perform_sql_to_document'")
    assert "not a table" in output


# --- perform_sql_to_document_records ---


def test_records_queries_the_chunks_table(tmp_path):
    tools, _ = build_tools(tmp_path)

    output = execute(
        tools,
        "perform_sql_to_document_records",
        sql="SELECT chunk_id FROM __chunks__ WHERE source_id = 'tbl-src'",
    )

    assert "sales-chunk" in output


def test_records_numbers_chunk_rows_so_they_are_citable(tmp_path):
    tools, _ = build_tools(tmp_path)

    output = execute(
        tools,
        "perform_sql_to_document_records",
        sql=(
            "SELECT chunk_id, origin_source_id, chat_id FROM __chunks__ "
            "WHERE source_id = 'tbl-src'"
        ),
    )

    # A chunk-shaped row (chunk_id + origin_source_id) is numbered, so the
    # answer can cite it.
    assert "[1] " in output
    assert "sales-chunk" in output
    assert "tbl-src" in output


def test_records_rejects_non_select(tmp_path):
    tools, _ = build_tools(tmp_path)

    output = execute(
        tools,
        "perform_sql_to_document_records",
        sql="DELETE FROM __chunks__ WHERE 1 = 1",
    )

    assert output == "Error: only read-only SELECT queries are allowed."


def test_records_rejects_cte_wrapped_queries(tmp_path):
    tools, _ = build_tools(tmp_path)

    output = execute(
        tools,
        "perform_sql_to_document_records",
        sql="WITH x AS (SELECT 1) SELECT * FROM x",
    )

    assert output == "Error: only read-only SELECT queries are allowed."


def test_records_scoped_to_the_chat(tmp_path):
    tools, _ = build_tools(tmp_path, chat_id="chat-a")

    # A query that omits chat_id is rejected upfront with a clear message
    # instead of failing with a cryptic SQL error.
    output = execute(
        tools,
        "perform_sql_to_document_records",
        sql="SELECT chunk_id FROM __chunks__",
    )
    assert "must include chat_id" in output

    # Selecting chat_id lets the scoped query through.
    output = execute(
        tools,
        "perform_sql_to_document_records",
        sql="SELECT chunk_id, chat_id FROM __chunks__ WHERE source_id = 'tbl-src'",
    )
    assert "sales-chunk" in output
