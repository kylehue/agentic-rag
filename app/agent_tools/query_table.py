import sqlite3

import pandas as pd

from app.agent_tools.base import AgentTool
from app.agent_tools.common import (
    QUERY_TABLE_MAX_ROWS,
    object_schema,
    read_sheet,
    resolve_table_file,
)


class QueryTableTool(AgentTool):
    """Run a read-only SQL query against one stored table (pandas + sqlite).

    The one tool that reads a table's data: the model writes targeted SQL
    and receives at most QUERY_TABLE_MAX_ROWS of results.
    """

    @property
    def name(self) -> str:
        return "query_table"

    @property
    def description(self) -> str:
        return (
            "Run a read-only SQL query against one stored table, using pandas "
            "on an in-memory sqlite database. The table is available as a "
            "single table named `data`. Write targeted SQL (specific columns, "
            "WHERE filters, LIMIT, and aggregates, for example "
            "SELECT SUM(amount) FROM data) rather than SELECT * over a whole "
            "table; at most the first 100 result rows are returned."
        )

    @property
    def parameters(self) -> dict:
        return object_schema(
            {
                "table": {
                    "type": "string",
                    "description": "The table name, as shown by list_tables.",
                },
                "source_id": {
                    "type": "string",
                    "description": (
                        "Disambiguate when several sources store a table with "
                        "this name; as shown by list_tables."
                    ),
                },
                "sql": {
                    "type": "string",
                    "description": "A read-only, targeted SQL SELECT query.",
                },
            },
            ["table", "sql"],
        )

    def create_executor(self, rag_service):
        sql_storage = rag_service.sql_storage
        file_storage = rag_service.file_storage

        async def execute(arguments: dict) -> str:
            table_name = str(arguments.get("table", ""))
            sql = str(arguments.get("sql", ""))
            raw_source_id = arguments.get("source_id")
            source_id = str(raw_source_id) if raw_source_id else None
            file_path, file_bytes = await resolve_table_file(
                sql_storage, file_storage, table_name, source_id
            )
            dataframe = read_sheet(table_name, file_path, file_bytes)

            # Read-only by construction: the table lives in a throw-away
            # in-memory database, loaded and closed around one query.
            connection = sqlite3.connect(":memory:")
            try:
                dataframe.to_sql("data", connection, index=False)
                result = pd.read_sql_query(sql, connection)
            finally:
                connection.close()

            total_rows = len(result)
            truncated = total_rows > QUERY_TABLE_MAX_ROWS
            csv = result.head(QUERY_TABLE_MAX_ROWS).to_csv(index=False).rstrip()
            note = (
                f"\n(showing the first {QUERY_TABLE_MAX_ROWS} of {total_rows} rows)"
                if truncated
                else ""
            )
            return (
                f"Result of the SQL query against table '{table_name}' "
                f"(the table is named `data` in the query):\n{csv}{note}"
            )

        return execute
