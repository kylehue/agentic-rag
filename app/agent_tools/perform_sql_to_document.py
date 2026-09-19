from app.agent_tools.base import AgentTool
from app.agent_tools.common import (
    SOURCE_ID_PARAMETER,
    object_schema,
    read_source_tables,
    resolve_table_document,
    run_table_sql,
)


class PerformSqlToDocumentTool(AgentTool):
    """Run read-only SQL over a source's table data (csv or workbook)."""

    @property
    def name(self) -> str:
        return "perform_sql_to_document"

    @property
    def description(self) -> str:
        return (
            "Run a read-only SQL query (sqlite dialect) against a source's "
            "tabular data. The source must be a csv or spreadsheet; its tables "
            "are loaded into a throw-away in-memory database, one table per "
            "sheet for a workbook. Each table is named by its table_name; quote "
            "a table or column name in double quotes if it contains spaces or "
            "is a reserved word. Add a LIMIT so the query fetches only the rows "
            "the answer actually needs, to avoid wasting tokens; omit the "
            "LIMIT only when you need every row of the result."
        )

    @property
    def parameters(self) -> dict:
        return object_schema(
            {
                "source_id": {
                    "type": "string",
                    "description": SOURCE_ID_PARAMETER,
                },
                "sql_query": {
                    "type": "string",
                    "description": (
                        "A read-only SQL SELECT query (sqlite dialect) over the "
                        "source's tables."
                    ),
                },
            },
            ["source_id", "sql_query"],
        )

    def create_executor(self, rag_service, chat_id=None):
        sql_storage = rag_service.sql_storage
        file_storage = rag_service.file_storage

        async def execute(arguments: dict) -> str:
            source_id = str(arguments.get("source_id", ""))
            sql = str(arguments.get("sql_query", ""))
            document = await resolve_table_document(sql_storage, source_id, chat_id)
            dataframes = await read_source_tables(file_storage, document)
            result = run_table_sql(dataframes, sql)

            tables = ", ".join(dataframes)
            return f"Tables: {tables}\n\n{result.to_csv(index=False).rstrip()}"

        return execute
