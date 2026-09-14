import json

from app.agent_tools.base import AgentTool
from app.agent_tools.common import object_schema, table_rows

# The metadata keys a table row is expected to carry, in display order.
METADATA_FIELDS = ("table_name", "row_count", "column_count", "source_page_number")


class ListTablesTool(AgentTool):
    """List the stored data tables (one chunk row per table)."""

    @property
    def name(self) -> str:
        return "list_tables"

    @property
    def description(self) -> str:
        return (
            "List the stored data tables, one line (JSON) per table: the "
            "source_id and chunk_id that address the table in inspect_table "
            "and query_table, plus the table's metadata fields (its name, "
            "row and column counts, and the source page number when the "
            "table is embedded in a parent document). Use it to find a "
            "table, then inspect_table to read its schema."
        )

    @property
    def parameters(self) -> dict:
        return object_schema({})

    def create_executor(self, rag_service, chat_id=None):
        sql_storage = rag_service.sql_storage

        async def execute(arguments: dict) -> str:
            rows = await table_rows(sql_storage, chat_id)
            if not rows:
                return "No tables are stored."

            lines = []
            for row in rows:
                metadata = row.get("metadata") or {}
                entry = {
                    "source_id": row["source_id"],
                    "chunk_id": row["chunk_id"],
                    **{
                        key: metadata[key]
                        for key in METADATA_FIELDS
                        if metadata.get(key) is not None
                    },
                }
                lines.append(json.dumps(entry))
            return "One line per table (JSON):\n" + "\n".join(lines)

        return execute
