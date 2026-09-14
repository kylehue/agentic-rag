import sqlite3

import pandas as pd

from app.agent_tools.base import AgentTool
from app.agent_tools.common import (
    CHUNK_ID_DESCRIPTION,
    QUERY_TABLE_MAX_ROWS,
    SOURCE_ID_DESCRIPTION,
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
                "source_id": {"type": "string", "description": SOURCE_ID_DESCRIPTION},
                "chunk_id": {"type": "string", "description": CHUNK_ID_DESCRIPTION},
                "sql": {
                    "type": "string",
                    "description": "A read-only, targeted SQL SELECT query.",
                },
            },
            ["source_id", "sql"],
        )

    def create_executor(self, rag_service, chat_id=None):
        sql_storage = rag_service.sql_storage
        file_storage = rag_service.file_storage

        async def execute(arguments: dict) -> str:
            source_id = str(arguments.get("source_id", ""))
            raw_chunk_id = arguments.get("chunk_id")
            chunk_id = str(raw_chunk_id) if raw_chunk_id else None
            sql = str(arguments.get("sql", ""))
            file_path, file_bytes, sheet = await resolve_table_file(
                sql_storage, file_storage, source_id, chunk_id, chat_id
            )
            dataframe = read_sheet(sheet, file_path, file_bytes)

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
                f"Result of the SQL query against table '{sheet or source_id}' "
                f"(the table is named `data` in the query):\n{csv}{note}"
            )

        return execute
