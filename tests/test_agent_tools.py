import asyncio
import io

from app.agent_tools import (
    InspectTableTool,
    ListTablesTool,
    QueryChunksTool,
    QueryTableTool,
    SearchDocumentTool,
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

CSV_BYTES = b"region,amount\nnorth,10\nsouth,25\n"
COLLIDING_CSV_BYTES = b"region,amount\nwest,5\n"
LEGACY_CSV_BYTES = b"name,qty\napple,3\n"


def workbook_bytes() -> bytes:
    """A two-sheet xlsx in memory: one stored table per sheet, both under a
    single source (the workbook file)."""
    import pandas as pd

    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer) as writer:
        pd.DataFrame({"region": ["west", "east"], "amount": [5, 7]}).to_excel(
            writer, sheet_name="Sheet A", index=False
        )
        pd.DataFrame({"name": ["apple"], "qty": [3]}).to_excel(
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


def build_tools(
    tmp_path,
    *,
    retriever_chunks=None,
    add_colliding_table=False,
    add_workbook=False,
    add_schemaless_table=False,
    chat_id=None,
):
    """Real local storages (with the system tables) plus the RAG tools.

    `add_colliding_table` stores a second source (another chat) with a
    table of the same name. `add_workbook` stores a two-sheet xlsx under
    one source, to exercise chunk_id selection among a source's tables.
    `add_schemaless_table` stores a table whose metadata predates schema
    storage, to exercise the inspect_table fallback. `chat_id` scopes the
    tools to one chat.
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

    async def setup():
        await service.initialize()
        doc_path = await file_storage.upload(
            file=io.BytesIO(b"The quarterly report body."),
            file_filename="report.txt",
            file_content_type="text/plain",
        )
        await sql_storage.upsert(
            settings.DOCUMENT_METADATA_TABLE_NAME,
            [
                {
                    "source_id": "doc-src",
                    "file_path": doc_path,
                    "file_content_type": "text/plain",
                    "file_filename": "report.txt",
                    "file_orig_filename": "report.txt",
                    "is_origin": True,
                }
            ],
        )
        path = await file_storage.upload(
            file=io.BytesIO(CSV_BYTES),
            file_filename="sales.csv",
            file_content_type="text/csv",
        )
        await sql_storage.upsert(
            settings.DOCUMENT_METADATA_TABLE_NAME,
            [
                {
                    "source_id": "tbl-src",
                    "file_path": path,
                    "file_content_type": "text/csv",
                    "file_filename": "sales.csv",
                    "file_orig_filename": "sales.csv",
                    "is_origin": True,
                }
            ],
        )
        await sql_storage.upsert(
            settings.CHUNK_TABLE_NAME,
            [
                {
                    "chunk_id": "text-chunk",
                    "source_id": "doc-src",
                    "parent_source_id": None,
                    "origin_source_id": "doc-src",
                    "plugin": "text",
                    "text": "The quarterly report body.",
                    "metadata": {"source_page_number": 1},
                    "chat_id": "chat-a",
                },
                {
                    "chunk_id": "table-chunk",
                    "source_id": "tbl-src",
                    "parent_source_id": None,
                    "origin_source_id": "tbl-src",
                    "plugin": "table",
                    "text": "Table: sales ...",
                    "metadata": {
                        "table_name": "sales",
                        "schema": [
                            {"name": "region", "type": "TEXT"},
                            {"name": "amount", "type": "INTEGER"},
                        ],
                        "row_count": 2,
                        "column_count": 2,
                    },
                    "chat_id": "chat-a",
                },
            ],
        )
        if add_colliding_table:
            path2 = await file_storage.upload(
                file=io.BytesIO(COLLIDING_CSV_BYTES),
                file_filename="sales_q4.csv",
                file_content_type="text/csv",
            )
            await sql_storage.upsert(
                settings.DOCUMENT_METADATA_TABLE_NAME,
                [
                    {
                        "source_id": "tbl-src-2",
                        "file_path": path2,
                        "file_content_type": "text/csv",
                        "file_filename": "sales_q4.csv",
                        "file_orig_filename": "sales_q4.csv",
                        "is_origin": True,
                    }
                ],
            )
            await sql_storage.upsert(
                settings.CHUNK_TABLE_NAME,
                [
                    {
                        "chunk_id": "table-chunk-2",
                        "source_id": "tbl-src-2",
                        "parent_source_id": None,
                        "origin_source_id": "tbl-src-2",
                        "plugin": "table",
                        "text": "Table: sales ...",
                        "metadata": {
                            "table_name": "sales",
                            "schema": [
                                {"name": "region", "type": "TEXT"},
                                {"name": "amount", "type": "INTEGER"},
                            ],
                            "row_count": 1,
                            "column_count": 2,
                        },
                        "chat_id": "chat-b",
                    }
                ],
            )
        if add_workbook:
            path_wb = await file_storage.upload(
                file=io.BytesIO(workbook_bytes()),
                file_filename="workbook.xlsx",
                file_content_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
            await sql_storage.upsert(
                settings.DOCUMENT_METADATA_TABLE_NAME,
                [
                    {
                        "source_id": "wb-src",
                        "file_path": path_wb,
                        "file_content_type": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        "file_filename": "workbook.xlsx",
                        "file_orig_filename": "workbook.xlsx",
                        "is_origin": True,
                    }
                ],
            )
            await sql_storage.upsert(
                settings.CHUNK_TABLE_NAME,
                [
                    {
                        "chunk_id": "wb-chunk-a",
                        "source_id": "wb-src",
                        "parent_source_id": None,
                        "origin_source_id": "wb-src",
                        "plugin": "table",
                        "text": "Table: Sheet A ...",
                        "metadata": {
                            "table_name": "Sheet A",
                            "schema": [
                                {"name": "region", "type": "TEXT"},
                                {"name": "amount", "type": "INTEGER"},
                            ],
                            "row_count": 2,
                            "column_count": 2,
                        },
                        "chat_id": "chat-a",
                    },
                    {
                        "chunk_id": "wb-chunk-b",
                        "source_id": "wb-src",
                        "parent_source_id": None,
                        "origin_source_id": "wb-src",
                        "plugin": "table",
                        "text": "Table: Sheet B ...",
                        "metadata": {
                            "table_name": "Sheet B",
                            "schema": [
                                {"name": "name", "type": "TEXT"},
                                {"name": "qty", "type": "INTEGER"},
                            ],
                            "row_count": 1,
                            "column_count": 2,
                        },
                        "chat_id": "chat-a",
                    },
                ],
            )
        if add_schemaless_table:
            path3 = await file_storage.upload(
                file=io.BytesIO(LEGACY_CSV_BYTES),
                file_filename="legacy.csv",
                file_content_type="text/csv",
            )
            await sql_storage.upsert(
                settings.DOCUMENT_METADATA_TABLE_NAME,
                [
                    {
                        "source_id": "legacy-src",
                        "file_path": path3,
                        "file_content_type": "text/csv",
                        "file_filename": "legacy.csv",
                        "file_orig_filename": "legacy.csv",
                        "is_origin": True,
                    }
                ],
            )
            await sql_storage.upsert(
                settings.CHUNK_TABLE_NAME,
                [
                    {
                        "chunk_id": "legacy-chunk",
                        "source_id": "legacy-src",
                        "parent_source_id": None,
                        "origin_source_id": "legacy-src",
                        "plugin": "table",
                        "text": "Table: legacy ...",
                        # No schema/description/row_count: pre-metadata table.
                        "metadata": {"table_name": "legacy"},
                        "chat_id": "chat-a",
                    }
                ],
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
        ListTablesTool(),
        InspectTableTool(),
        QueryTableTool(),
        QueryChunksTool(),
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
    # Each chunk carries its source and chunk ids; the tool knows nothing
    # about the citation format (that is the agent service's concern).
    assert "[1] source=s1 chunk=c1 first evidence" in output
    assert "[2] source=s1 chunk=c2 second evidence" in output


def test_search_documents_reports_no_evidence(tmp_path):
    tools, _ = build_tools(tmp_path)

    assert execute(tools, "search_documents", query="anything") == "No evidence found."


# --- list_tables ---


def test_list_tables_lists_the_stored_tables(tmp_path):
    tools, _ = build_tools(tmp_path)

    output = execute(tools, "list_tables")

    assert '"table_name": "sales"' in output
    assert '"source_id": "tbl-src"' in output
    assert '"chunk_id": "table-chunk"' in output
    assert '"row_count": 2' in output
    # The text chunk is not a table.
    assert "text-chunk" not in output


# --- inspect_table ---


def test_inspect_table_reads_schema_and_shape_from_metadata(tmp_path):
    tools, _ = build_tools(tmp_path)

    output = execute(tools, "inspect_table", source_id="tbl-src")

    # Shape and schema come from stored metadata...
    assert "2 rows x 2 columns" in output
    assert "region: TEXT" in output
    assert "amount: INTEGER" in output
    # ...and no data rows are leaked (that is query_table's job). The
    # description is not shown either — it lives in the chunk's text, which
    # search_documents returns.
    assert "north" not in output
    assert "south" not in output
    assert "Description:" not in output


def test_inspect_table_falls_back_to_inferring_schema(tmp_path):
    tools, _ = build_tools(tmp_path, add_schemaless_table=True)

    output = execute(tools, "inspect_table", source_id="legacy-src")

    # No stored schema, so it is inferred from a bounded peek at the sheet;
    # the single data row means the row count is exact.
    assert "1 rows x 2 columns" in output
    assert "name: TEXT" in output
    assert "qty: INTEGER" in output


def test_inspect_table_unknown_source_is_an_error_string(tmp_path):
    tools, _ = build_tools(tmp_path)

    output = execute(tools, "inspect_table", source_id="missing")

    assert output.startswith("Error in tool 'inspect_table'")
    assert "Unknown source 'missing'" in output


def test_inspect_table_a_non_table_source_is_an_error(tmp_path):
    tools, _ = build_tools(tmp_path)

    # doc-src is the quarterly report, not a table.
    output = execute(tools, "inspect_table", source_id="doc-src")

    assert output.startswith("Error in tool 'inspect_table'")
    assert "isn't a table" in output


# --- query_table ---


def test_query_table_computes_with_sql(tmp_path):
    tools, _ = build_tools(tmp_path)

    output = execute(
        tools,
        "query_table",
        source_id="tbl-src",
        sql="SELECT SUM(amount) AS total FROM data",
    )

    assert "total" in output
    assert "35" in output


def test_query_table_filters_rows(tmp_path):
    tools, _ = build_tools(tmp_path)

    output = execute(
        tools,
        "query_table",
        source_id="tbl-src",
        sql="SELECT region FROM data WHERE amount > 15",
    )

    assert "south" in output
    assert "north" not in output


def test_query_table_invalid_sql_is_an_error_string(tmp_path):
    tools, _ = build_tools(tmp_path)

    output = execute(
        tools,
        "query_table",
        source_id="tbl-src",
        sql="SELECT * FROM nonexistent",
    )

    assert output.startswith("Error in tool 'query_table'")


# --- multi-table sources (a source storing several tables) ---


def test_multi_table_source_requires_the_chunk_id(tmp_path):
    tools, _ = build_tools(tmp_path, add_workbook=True)

    output = execute(tools, "inspect_table", source_id="wb-src")

    assert output.startswith("Error in tool 'inspect_table'")
    assert "several tables" in output
    # Both of the source's tables are named so the model can pick one.
    assert "Sheet A" in output and "Sheet B" in output
    assert "chunk_id" in output


def test_chunk_id_selects_one_table_of_a_source(tmp_path):
    tools, _ = build_tools(tmp_path, add_workbook=True)

    output = execute(
        tools, "inspect_table", source_id="wb-src", chunk_id="wb-chunk-b"
    )

    assert "1 rows x 2 columns" in output
    assert "name: TEXT" in output
    assert "qty: INTEGER" in output


def test_query_table_targets_one_sheet_of_a_source(tmp_path):
    tools, _ = build_tools(tmp_path, add_workbook=True)

    # Sheet A (west 5, east 7) totals 12; Sheet B has no `amount` column at
    # all, so a wrong selection cannot silently answer the query.
    output = execute(
        tools,
        "query_table",
        source_id="wb-src",
        chunk_id="wb-chunk-a",
        sql="SELECT SUM(amount) AS total FROM data",
    )

    assert "total" in output
    assert "12" in output
    assert "Sheet A" in output


def test_wrong_chunk_id_for_a_source_is_an_error(tmp_path):
    tools, _ = build_tools(tmp_path, add_workbook=True)

    output = execute(
        tools, "inspect_table", source_id="wb-src", chunk_id="nope"
    )

    assert output.startswith("Error in tool 'inspect_table'")
    assert "no table with chunk_id 'nope'" in output


# --- same-named tables in different sources ---


def test_same_named_tables_are_addressed_by_their_own_source(tmp_path):
    tools, _ = build_tools(tmp_path, add_colliding_table=True)

    # Both sources store a table named 'sales'; the source id is the
    # address, so there is nothing to disambiguate.
    output_a = execute(tools, "inspect_table", source_id="tbl-src")
    output_b = execute(tools, "inspect_table", source_id="tbl-src-2")

    assert "2 rows x 2 columns" in output_a
    assert "1 rows x 2 columns" in output_b


# --- query_chunks ---


def test_query_chunks_reads_the_document_database(tmp_path):
    tools, _ = build_tools(tmp_path)

    output = execute(
        tools,
        "query_chunks",
        sql="SELECT chunk_id, plugin FROM __chunks__ WHERE plugin = 'table'",
    )

    assert "table-chunk" in output
    assert "text-chunk" not in output


def test_query_chunks_rejects_non_select(tmp_path):
    tools, _ = build_tools(tmp_path)

    output = execute(
        tools, "query_chunks", sql="DELETE FROM __chunks__ WHERE 1 = 1"
    )

    assert output == "Error: only read-only SELECT queries are allowed."


def test_query_chunks_rejects_cte_wrapped_queries(tmp_path):
    tools, _ = build_tools(tmp_path)

    output = execute(
        tools,
        "query_chunks",
        sql="WITH x AS (SELECT 1 AS n) SELECT * FROM x",
    )

    # A CTE can wrap a DELETE in SQLite, so WITH is not a SELECT alias; the
    # guard matches the storage layer's backstop.
    assert output == "Error: only read-only SELECT queries are allowed."


# --- tool definitions ---


def rag_tools():
    return [
        SearchDocumentTool(),
        ListTablesTool(),
        InspectTableTool(),
        QueryTableTool(),
        QueryChunksTool(),
    ]


def test_tool_definitions_carry_name_description_and_parameters():
    tools = rag_tools()

    assert {tool.name for tool in tools} == {
        "search_documents",
        "list_tables",
        "inspect_table",
        "query_table",
        "query_chunks",
    }
    assert QueryTableTool().parameters["required"] == ["source_id", "sql"]


def test_outline_summarizes_each_tool_from_its_definition():
    outline = tools_outline(rag_tools())

    # Every tool, with its description...
    for name in (
        "search_documents",
        "list_tables",
        "inspect_table",
        "query_table",
        "query_chunks",
    ):
        assert f"`{name}`" in outline
    assert "Search the ingested document corpus" in outline
    # ...and a compact parameter summary derived from its JSON schema.
    assert "Parameters: query (string, required)" in outline
    assert (
        "Parameters: source_id (string, required), chunk_id (string, optional), "
        "sql (string, required)" in outline
    )
    # Four of the five tools take parameters; list_tables gets no line.
    assert outline.count("Parameters: ") == 4


def test_unknown_tool_returns_an_error_string(tmp_path):
    executors, _ = build_tools(tmp_path)

    assert execute(executors, "nope") == "Error: unknown tool 'nope'."


# --- chat scoping ---
# chat-a holds the default fixtures; chat-b the colliding table; chat-empty
# nothing. Separate storage dirs because the setup is not idempotent for a
# second call on the same path.


def test_list_tables_scoped_to_the_chat(tmp_path):
    executors_a, _ = build_tools(
        tmp_path / "a", add_colliding_table=True, chat_id="chat-a"
    )
    executors_b, _ = build_tools(
        tmp_path / "b", add_colliding_table=True, chat_id="chat-b"
    )
    executors_none, _ = build_tools(
        tmp_path / "c", add_colliding_table=True, chat_id="chat-empty"
    )

    output_a = execute(executors_a, "list_tables")
    assert '"source_id": "tbl-src"' in output_a
    assert "tbl-src-2" not in output_a
    # The other chat sees its own table of the same name...
    output_b = execute(executors_b, "list_tables")
    assert '"source_id": "tbl-src-2"' in output_b
    assert '"source_id": "tbl-src"' not in output_b
    # ...and a chat without tables sees none.
    assert execute(executors_none, "list_tables") == "No tables are stored."


def test_inspect_table_scoped_outside_the_chat(tmp_path):
    executors, _ = build_tools(tmp_path, chat_id="chat-empty")

    # tbl-src's table lives in chat-a, outside this chat.
    output = execute(executors, "inspect_table", source_id="tbl-src")
    assert output.startswith("Error in tool 'inspect_table'")
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
        "[1] source=s1 chunk=c1 evidence"
        in execute(executors_a, "search_documents", query="q")
    )
    assert "chunk=c2" not in execute(executors_a, "search_documents", query="q")
    assert (
        execute(executors_none, "search_documents", query="q") == "No evidence found."
    )
    # The chat id reached the retrieval path as the where condition.
    assert retriever_a.wheres == [{"chat_id": "chat-a"}, {"chat_id": "chat-a"}]


def test_query_chunks_scoped_to_the_chat(tmp_path):
    executors, _ = build_tools(
        tmp_path, add_colliding_table=True, chat_id="chat-a"
    )

    output = execute(
        executors,
        "query_chunks",
        sql="SELECT source_id, chat_id FROM __chunks__ WHERE plugin = 'table'",
    )
    # Only this chat's rows come back...
    assert "tbl-src" in output
    assert "tbl-src-2" not in output
    # ...and a query without the chat_id column cannot bypass the scope.
    bypass = execute(
        executors, "query_chunks", sql="SELECT chunk_id FROM __chunks__"
    )
    assert bypass.startswith("Error in tool 'query_chunks'")
