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


class FixedRetriever(Retriever):
    def __init__(self, chunks: list[RetrievedChunk]) -> None:
        self._chunks = chunks
        self.queries: list[str] = []

    async def retrieve(self, user_query) -> list[RetrievedChunk]:
        self.queries.append(user_query)
        return self._chunks


def make_chunk(chunk_id: str, plugin: str, text: str, **metadata) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        source_id="s1",
        origin_source_id="s1",
        plugin=plugin,
        text=text,
        score=0.9,
        metadata=metadata,
    )


def build_tools(
    tmp_path,
    *,
    retriever_chunks=None,
    add_colliding_table=False,
    add_schemaless_table=False,
):
    """Real local storages (with the system tables) plus the RAG tools.

    `add_colliding_table` stores a second source file whose table has the
    same `table_name`, to exercise name disambiguation. `add_schemaless_table`
    stores a table whose metadata predates schema storage, to exercise the
    inspect_table fallback.
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
                    }
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
    executors = {tool.name: tool.create_executor(rag_service) for tool in tools}
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

    output = execute(tools, "inspect_table", table="sales")

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

    output = execute(tools, "inspect_table", table="legacy")

    # No stored schema, so it is inferred from a bounded peek at the sheet;
    # the single data row means the row count is exact.
    assert "1 rows x 2 columns" in output
    assert "name: TEXT" in output
    assert "qty: INTEGER" in output


def test_inspect_table_unknown_name_is_an_error_string(tmp_path):
    tools, _ = build_tools(tmp_path)

    output = execute(tools, "inspect_table", table="missing")

    assert output.startswith("Error in tool 'inspect_table'")
    assert "No table named 'missing'" in output


# --- query_table ---


def test_query_table_computes_with_sql(tmp_path):
    tools, _ = build_tools(tmp_path)

    output = execute(
        tools,
        "query_table",
        table="sales",
        sql="SELECT SUM(amount) AS total FROM data",
    )

    assert "total" in output
    assert "35" in output


def test_query_table_filters_rows(tmp_path):
    tools, _ = build_tools(tmp_path)

    output = execute(
        tools,
        "query_table",
        table="sales",
        sql="SELECT region FROM data WHERE amount > 15",
    )

    assert "south" in output
    assert "north" not in output


def test_query_table_invalid_sql_is_an_error_string(tmp_path):
    tools, _ = build_tools(tmp_path)

    output = execute(
        tools, "query_table", table="sales", sql="SELECT * FROM nonexistent"
    )

    assert output.startswith("Error in tool 'query_table'")


# --- colliding table names ---


def test_colliding_table_names_are_ambiguous_until_disambiguated(tmp_path):
    tools, _ = build_tools(tmp_path, add_colliding_table=True)

    output = execute(tools, "inspect_table", table="sales")

    assert output.startswith("Error in tool 'inspect_table'")
    assert "ambiguous" in output
    # Both stored sources are named so the model can disambiguate.
    assert "tbl-src" in output and "tbl-src-2" in output


def test_source_id_disambiguates_a_colliding_table_name(tmp_path):
    tools, _ = build_tools(tmp_path, add_colliding_table=True)

    output = execute(
        tools,
        "query_table",
        table="sales",
        source_id="tbl-src-2",
        sql="SELECT SUM(amount) AS total FROM data",
    )

    # The second source's table (west,5), not the first's (total 35).
    assert "total" in output
    assert output.rstrip().endswith("5")
    assert "35" not in output


def test_unknown_source_id_for_a_colliding_table_name_is_an_error(tmp_path):
    tools, _ = build_tools(tmp_path, add_colliding_table=True)

    output = execute(tools, "inspect_table", table="sales", source_id="nope")

    assert output.startswith("Error in tool 'inspect_table'")
    assert "stored under source 'nope'" in output


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
    assert QueryTableTool().parameters["required"] == ["table", "sql"]


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
        "Parameters: table (string, required), source_id (string, optional), "
        "sql (string, required)" in outline
    )
    # Four of the five tools take parameters; list_tables gets no line.
    assert outline.count("Parameters: ") == 4


def test_unknown_tool_returns_an_error_string(tmp_path):
    executors, _ = build_tools(tmp_path)

    assert execute(executors, "nope") == "Error: unknown tool 'nope'."
