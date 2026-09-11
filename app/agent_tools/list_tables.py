import json

from app.agent.tools import AgentTool
from app.agent_tools.common import object_schema, table_rows
from app.store_sql.base import SqlStorage


def make_list_tables_tool(sql_storage: SqlStorage) -> AgentTool:
    """List the stored data tables (one chunk row per table)."""

    async def execute(arguments: dict) -> str:
        rows = await table_rows(sql_storage)
        if not rows:
            return "No tables are stored."

        lines = [
            json.dumps(
                {
                    "table_name": (row.get("metadata") or {}).get("table_name"),
                    "source_id": row["source_id"],
                    "chunk_id": row["chunk_id"],
                    "origin_source_id": row["origin_source_id"],
                    "row_count": (row.get("metadata") or {}).get("row_count"),
                    "column_count": (row.get("metadata") or {}).get("column_count"),
                    "source_page_number": (row.get("metadata") or {}).get(
                        "source_page_number"
                    ),
                }
            )
            for row in rows
        ]
        return "One line per table (JSON):\n" + "\n".join(lines)

    return AgentTool(
        name="list_tables",
        description=(
            "List the stored data tables: table name, source id, chunk id, "
            "row count, column count, and the source page number when the "
            "table is embedded in a parent document. Use it to find a table, "
            "then inspect_table to read its schema."
        ),
        parameters=object_schema({}),
        execute=execute,
    )
