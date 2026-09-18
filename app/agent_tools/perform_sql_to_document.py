from app.agent_tools.base import AgentTool
from app.agent_tools.common import (
    SOURCE_ID_PARAMETER,
    SQL_MAX_ROWS,
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
            "is a reserved word. At most the first "
            f"{SQL_MAX_ROWS} result rows are returned."
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

            total_rows = len(result)
            truncated = total_rows > SQL_MAX_ROWS
            csv = result.head(SQL_MAX_ROWS).to_csv(index=False).rstrip()
            note = (
                f"\n(showing the first {SQL_MAX_ROWS} of {total_rows} rows)"
                if truncated
                else ""
            )
            tables = ", ".join(dataframes)
            return f"Tables: {tables}\n\n{csv}{note}"

        return execute
