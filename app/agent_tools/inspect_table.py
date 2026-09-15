import json

from app.agent_tools.base import AgentTool
from app.agent_tools.common import SOURCE_ID_DESCRIPTION, object_schema, resolve_csv_table


class InspectTableTool(AgentTool):
    """Show one stored csv table's full record (metadata + chunk info)."""

    @property
    def name(self) -> str:
        return "inspect_table"

    @property
    def description(self) -> str:
        return (
            "Show one stored table's full record: its entire metadata "
            "(schema, row and column counts, ...) and the chunk's info "
            "(source_id, chunk_id, parent_source_id, origin_source_id) and "
            "text (the table's description and a sample). Only tables stored "
            "as csv files can be inspected."
        )

    @property
    def parameters(self) -> dict:
        return object_schema(
            {
                "source_id": {"type": "string", "description": SOURCE_ID_DESCRIPTION},
            },
            ["source_id"],
        )

    def create_executor(self, rag_service, chat_id=None):
        sql_storage = rag_service.sql_storage

        async def execute(arguments: dict) -> str:
            source_id = str(arguments.get("source_id", ""))
            row, _document = await resolve_csv_table(sql_storage, source_id, chat_id)
            payload = {
                "source_id": row["source_id"],
                "chunk_id": row["chunk_id"],
                "parent_source_id": row["parent_source_id"],
                "origin_source_id": row["origin_source_id"],
                "text": row["text"],
                "metadata": row.get("metadata") or {},
            }
            return json.dumps(payload, default=str)

        return execute
