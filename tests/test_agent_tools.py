import asyncio
import io
import json

from app.agent_tools import (
    InspectTableRelationshipsTool,
    InspectTableTool,
    SearchDocumentTool,
    SqlQueryDocumentsTool,
    SqlQueryTableTool,
    tools_outline,
)
from app.core.config import settings
from app.models.chunk import RetrievedChunk
from app.plugin.registry import PluginRegistry
from app.retrievers.base import Retriever
from app.services.ingestion import IngestionService
from app.services.rag_agent import execute_tool
from app.store_file.local import LocalFileStorage
from app.store_sql.local import LocalSqlStorage

from fakes import FakeEmbedder, FakeLLM, FakeVectorStorage, build_rag_service

SALES_CSV = b"region,amount\nnorth,10\nsouth,25\n"
BIG_CSV = b"n\n1\n2\n3\n4\n5\n6\n"
EMBEDDED_SALES_CSV = b"region,amount\nwest,5\neast,7\n"
EMBEDDED_UNITS_CSV = b"region,units\nwest,2\neast,3\n"


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
    """Real local storages (with the system tables) plus the RAG tools.

    The corpus (all in chat-a): a text document, a directly ingested csv
    table, a big csv table, a two-sheet workbook, and a pdf whose two
    embedded tables are stored as csvs sharing the pdf as their origin.
    `chat_id` scopes the tools to one chat.
    """
    retriever = FixedRetriever(retriever_chunks or [])
    sql_storage = LocalSqlStorage(storage_dir=tmp_path / "sql")
    file_storage = LocalFileStorage(storage_dir=tmp_path / "file")

    service = IngestionService(
        registry=PluginRegistry(),
        llm=FakeLLM(),
        embedder=FakeEmbedder(),
        vector_storage=FakeVectorStorage(),
        sql_storage=sql_storage,
        file_storage=file_storage,
    )

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

    async def setup():
        await service.initialize()

        async def store_document(source_id, filename, content_type, file_bytes):
            path = await file_storage.upload(
                file=io.BytesIO(file_bytes),
                file_filename=filename,
                file_content_type=content_type,
            )
            await sql_storage.upsert(
                settings.DOCUMENT_METADATA_TABLE_NAME,
                [
                    {
                        "source_id": source_id,
                        "file_path": path,
                        "file_content_type": content_type,
                        "file_filename": filename,
                        "file_orig_filename": filename,
                        "is_origin": True,
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
                settings.CHUNK_TABLE_NAME,
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

        # A two-sheet workbook (one source, several tables).
        await store_document(
            "wb-src", "book.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            b"fake-xlsx-bytes",
        )
        await store_chunk(
            "sheet-a-chunk", "wb-src", "table", "Table: Sheet A ...",
            table_metadata("Sheet A", SALES_SCHEMA, 2),
        )
        await store_chunk(
            "sheet-b-chunk", "wb-src", "table", "Table: Sheet B ...",
            table_metadata(
                "Sheet B",
                [
                    {"name": "name", "type": "TEXT"},
                    {"name": "qty", "type": "INTEGER"},
                ],
                1,
            ),
        )

        # A pdf with two embedded tables (csvs sharing the pdf origin).
        await store_document(
            "pdf-src", "report.pdf", "application/pdf", b"fake-pdf-bytes"
        )
        await store_chunk(
            "pdf-text-chunk", "pdf-src", "text", "The report body.",
            {"source_page_number": 2},
        )
        await store_document(
            "emb-1", "report_table_1.csv", "text/csv", EMBEDDED_SALES_CSV
        )
        await store_chunk(
            "emb-1-chunk", "emb-1", "table", "Table: report_table_1 ...",
            table_metadata("report_table_1", SALES_SCHEMA, 2),
            origin_source_id="pdf-src",
            parent_source_id="pdf-src",
        )
        await store_document(
            "emb-2", "report_table_2.csv", "text/csv", EMBEDDED_UNITS_CSV
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
            origin_source_id="pdf-src",
            parent_source_id="pdf-src",
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
        InspectTableTool(),
        InspectTableRelationshipsTool(),
        SqlQueryTableTool(),
        SqlQueryDocumentsTool(),
    ]
    executors = {
        tool.name: tool.create_executor(rag_service, chat_id) for tool in tools
    }
    return executors, retriever


def execute(executors, name: str, **arguments) -> str:
    return asyncio.run(execute_tool(executors, name, arguments))


# --- search_documents ---


def test_search_documents_returns_ranked_evidence(tmp_path):
    tools, retriever = build_tools(
        tmp_path,
        retriever_chunks=[
            make_chunk("c1", "text", "first evidence"),
            make_chunk("c2", "table", "second evidence", table_name="sales"),
        ],
    )

    output = execute(tools, "search_documents", query="evidence?")

    assert retriever.queries == ["evidence?"]
    # Each chunk carries its origin source id and chunk id (the ids the model
    # cites); the tool knows nothing about the citation format itself.
    assert "[1] origin=s1 chunk=c1 first evidence" in output
    assert "[2] origin=s1 chunk=c2 second evidence" in output


def test_search_documents_reports_no_evidence(tmp_path):
    tools, _ = build_tools(tmp_path)

    assert execute(tools, "search_documents", query="anything") == "No evidence found."


# --- inspect_table ---


def test_inspect_table_returns_the_full_record(tmp_path):
    tools, _ = build_tools(tmp_path)

    payload = json.loads(execute(tools, "inspect_table", source_id="tbl-src"))

    assert payload["source_id"] == "tbl-src"
    assert payload["chunk_id"] == "sales-chunk"
    assert payload["parent_source_id"] is None
    assert payload["origin_source_id"] == "tbl-src"
    assert payload["text"] == "Table: sales ..."
    # The entire metadata, not a selection of it.
    assert payload["metadata"]["table_name"] == "sales"
    assert payload["metadata"]["schema"] == [
        {"name": "region", "type": "TEXT"},
        {"name": "amount", "type": "INTEGER"},
    ]
    assert payload["metadata"]["row_count"] == 2
    assert payload["metadata"]["column_count"] == 2


def test_inspect_table_rejects_a_non_csv_source(tmp_path):
    tools, _ = build_tools(tmp_path)

    output = execute(tools, "inspect_table", source_id="wb-src")

    assert output.startswith("Error in tool 'inspect_table'")
    assert "not a csv table" in output


def test_inspect_table_rejects_a_source_without_tables(tmp_path):
    tools, _ = build_tools(tmp_path)

    # A csv source with no table chunk...
    output = execute(tools, "inspect_table", source_id="empty-src")
    assert output.startswith("Error in tool 'inspect_table'")
    assert "stores no table" in output
    # ...and a non-csv document is rejected before any table lookup.
    output = execute(tools, "inspect_table", source_id="doc-src")
    assert "not a csv table" in output


def test_inspect_table_rejects_an_unknown_source(tmp_path):
    tools, _ = build_tools(tmp_path)

    output = execute(tools, "inspect_table", source_id="nope")

    assert output.startswith("Error in tool 'inspect_table'")
    assert "Unknown source 'nope'" in output


# --- inspect_table_relationships ---


def test_relationships_list_the_sibling_tables_of_the_origin(tmp_path):
    tools, _ = build_tools(tmp_path)

    output = execute(tools, "inspect_table_relationships", source_id="emb-1")

    # The sibling embedded table (same pdf origin), not itself.
    assert "emb-2" in output
    assert "emb-1" not in output
    line = next(json.loads(l) for l in output.splitlines() if l.startswith("{"))
    assert line["source_id"] == "emb-2"
    assert line["chunk_id"] == "emb-2-chunk"
    assert "Table: report_table_2" in line["description"]
    assert line["schema"] == "region: TEXT, units: INTEGER"


def test_relationships_of_a_workbook_source_list_its_sheets(tmp_path):
    tools, _ = build_tools(tmp_path)

    output = execute(tools, "inspect_table_relationships", source_id="wb-src")

    # The workbook's sheets are each other's possible relationships.
    assert "Sheet A" in output and "Sheet B" in output
    assert output.count("wb-src") >= 2


def test_relationships_with_no_siblings(tmp_path):
    tools, _ = build_tools(tmp_path)

    output = execute(tools, "inspect_table_relationships", source_id="tbl-src")

    assert "No other tables share this table's origin document." in output


def test_relationships_reject_non_table_sources(tmp_path):
    tools, _ = build_tools(tmp_path)

    output = execute(tools, "inspect_table_relationships", source_id="doc-src")
    assert output.startswith("Error in tool 'inspect_table_relationships'")
    assert "stores no table" in output

    output = execute(tools, "inspect_table_relationships", source_id="nope")
    assert "Unknown source 'nope'" in output


# --- sql_query_table ---


def test_sql_query_table_computes_with_sql(tmp_path):
    tools, _ = build_tools(tmp_path)

    output = execute(
        tools,
        "sql_query_table",
        source_id="tbl-src",
        sql="SELECT SUM(amount) AS total FROM data",
    )

    assert "total" in output
    assert "35" in output


def test_sql_query_table_joins_related_tables(tmp_path):
    tools, _ = build_tools(tmp_path)

    output = execute(
        tools,
        "sql_query_table",
        source_id="emb-1",
        sql=(
            "SELECT related.units FROM data "
            "JOIN related ON data.region = related.region "
            "WHERE data.region = 'east'"
        ),
        table_relationships={"related": "emb-2"},
    )

    assert "units" in output
    assert "3" in output
    assert "related tables loaded as: related" in output


def test_sql_query_table_rejects_a_non_csv_source(tmp_path):
    tools, _ = build_tools(tmp_path)

    output = execute(
        tools, "sql_query_table", source_id="wb-src", sql="SELECT 1"
    )

    assert output.startswith("Error in tool 'sql_query_table'")
    assert "not a csv table" in output


def test_sql_query_table_rejects_a_non_csv_related_table(tmp_path):
    tools, _ = build_tools(tmp_path)

    output = execute(
        tools,
        "sql_query_table",
        source_id="tbl-src",
        sql="SELECT 1 FROM data",
        table_relationships={"r": "wb-src"},
    )

    assert output.startswith("Error in tool 'sql_query_table'")
    assert "not a csv table" in output


def test_sql_query_table_rejects_bad_aliases(tmp_path):
    tools, _ = build_tools(tmp_path)

    output = execute(
        tools,
        "sql_query_table",
        source_id="tbl-src",
        sql="SELECT 1 FROM data",
        table_relationships={"data": "emb-2"},
    )
    assert output.startswith("Error in tool 'sql_query_table'")
    assert "reserved" in output

    output = execute(
        tools,
        "sql_query_table",
        source_id="tbl-src",
        sql="SELECT 1 FROM data",
        table_relationships={"1bad": "emb-2"},
    )
    assert "Invalid table alias '1bad'" in output


def test_sql_query_table_caps_the_result_rows(tmp_path):
    tools, _ = build_tools(tmp_path)

    output = execute(
        tools, "sql_query_table", source_id="big-src", sql="SELECT n FROM data"
    )

    assert "showing the first 5 of 6 rows" in output
    assert "1" in output and "5" in output
    assert "\n6" not in output


# --- sql_query_documents ---


def test_sql_query_documents_reads_the_chunks_table(tmp_path):
    tools, _ = build_tools(tmp_path, chat_id="chat-a")

    output = execute(
        tools,
        "sql_query_documents",
        sql="SELECT chunk_id, chat_id FROM __chunks__ WHERE chat_id = 'chat-a'",
    )

    assert "sales-chunk" in output
    assert "text-chunk" in output
    # The scope wraps the query; rows from other chats never come back.
    assert "chat-b" not in output


def test_sql_query_documents_rejects_non_select(tmp_path):
    tools, _ = build_tools(tmp_path)

    output = execute(
        tools, "sql_query_documents", sql="DELETE FROM __chunks__ WHERE 1 = 1"
    )

    assert output == "Error: only read-only SELECT queries are allowed."


def test_sql_query_documents_rejects_cte_wrapped_queries(tmp_path):
    tools, _ = build_tools(tmp_path)

    output = execute(
        tools,
        "sql_query_documents",
        sql="WITH x AS (SELECT 1 AS n) SELECT * FROM x",
    )

    # A CTE can wrap a DELETE in SQLite, so WITH is not a SELECT alias; the
    # guard matches the storage layer's backstop.
    assert output == "Error: only read-only SELECT queries are allowed."


def test_sql_query_documents_scoped_to_the_chat(tmp_path):
    tools, _ = build_tools(tmp_path, chat_id="chat-a")

    output = execute(
        tools,
        "sql_query_documents",
        sql="SELECT chunk_id, chat_id FROM __chunks__ WHERE chat_id = 'chat-a'",
    )
    assert "sales-chunk" in output
    # ...and a query that does not select chat_id cannot bypass the scope.
    bypass = execute(tools, "sql_query_documents", sql="SELECT chunk_id FROM __chunks__")
    assert bypass.startswith("Error in tool 'sql_query_documents'")


# --- tool definitions ---


def rag_tools():
    return [
        SearchDocumentTool(),
        InspectTableTool(),
        InspectTableRelationshipsTool(),
        SqlQueryTableTool(),
        SqlQueryDocumentsTool(),
    ]


def test_tool_definitions_carry_name_description_and_parameters():
    tools = rag_tools()

    assert {tool.name for tool in tools} == {
        "search_documents",
        "inspect_table",
        "inspect_table_relationships",
        "sql_query_table",
        "sql_query_documents",
    }
    assert InspectTableTool().parameters["required"] == ["source_id"]
    assert SqlQueryTableTool().parameters["required"] == ["source_id", "sql"]


def test_outline_summarizes_each_tool_from_its_definition():
    outline = tools_outline(rag_tools())

    # Every tool, with its description...
    for name in (
        "search_documents",
        "inspect_table",
        "inspect_table_relationships",
        "sql_query_table",
        "sql_query_documents",
    ):
        assert f"`{name}`" in outline
    # ...and a compact parameter summary derived from its JSON schema.
    assert "Parameters: query (string, required)" in outline
    assert "Parameters: source_id (string, required)" in outline
    assert (
        "Parameters: source_id (string, required), sql (string, required), "
        "table_relationships (object, optional)" in outline
    )
    # All five tools take parameters.
    assert outline.count("Parameters: ") == 5


def test_unknown_tool_returns_an_error_string(tmp_path):
    executors, _ = build_tools(tmp_path)

    assert execute(executors, "nope") == "Error: unknown tool 'nope'."


# --- chat scoping ---
# The fixture corpus lives in chat-a; a chat-empty scope sees none of it.
# Separate storage dirs because the setup is not idempotent for a second
# call on the same path.


def test_inspect_table_scoped_outside_the_chat(tmp_path):
    executors, _ = build_tools(tmp_path / "a", chat_id="chat-empty")

    output = execute(executors, "inspect_table", source_id="tbl-src")
    assert output.startswith("Error in tool 'inspect_table'")
    assert "not in this chat" in output


def test_relationships_scoped_to_the_chat(tmp_path):
    executors, _ = build_tools(tmp_path / "a", chat_id="chat-empty")

    output = execute(executors, "inspect_table_relationships", source_id="emb-1")
    assert output.startswith("Error in tool 'inspect_table_relationships'")
    assert "not in this chat" in output


def test_search_documents_scoped_to_the_chat(tmp_path):
    chunks = [
        make_chunk("c1", "text", "evidence", chat_id="chat-a"),
        make_chunk("c2", "table", "more", chat_id="chat-b"),
    ]
    executors_a, retriever_a = build_tools(
        tmp_path / "a", chat_id="chat-a", retriever_chunks=chunks
    )
    executors_none, _ = build_tools(
        tmp_path / "n", chat_id="chat-empty", retriever_chunks=chunks
    )

    assert (
        "[1] origin=s1 chunk=c1 evidence"
        in execute(executors_a, "search_documents", query="q")
    )
    assert "chunk=c2" not in execute(executors_a, "search_documents", query="q")
    assert (
        execute(executors_none, "search_documents", query="q") == "No evidence found."
    )
    # The chat id reached the retrieval path as the where condition.
    assert retriever_a.wheres == [{"chat_id": "chat-a"}, {"chat_id": "chat-a"}]
