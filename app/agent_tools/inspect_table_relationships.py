import json

from app.agent_tools.base import AgentTool
from app.agent_tools.common import (
    SOURCE_ID_DESCRIPTION,
    format_schema,
    object_schema,
    resolve_table_source,
    table_chunks_with_origin,
)


class InspectTableRelationshipsTool(AgentTool):
    """Suggest the other tables a given table could be related to."""

    @property
    def name(self) -> str:
        return "inspect_table_relationships"

    @property
    def description(self) -> str:
        return (
            "Suggest other stored tables that could be related to a table: "
            "the tables that were ingested from the same origin document "
            "(for example the other tables embedded in the same document, or "
            "the other sheets of the same workbook). Returns each candidate's "
            "source_id, description, and schema."
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
            rows = await resolve_table_source(sql_storage, source_id, chat_id)
            origin = rows[0]["origin_source_id"]

            candidates = await table_chunks_with_origin(
                sql_storage, origin, chat_id
            )
            # For a single-table source, the candidates are its siblings;
            # for a multi-table source (a workbook), they are its sheets.
            if len(rows) == 1:
                candidates = [
                    row for row in candidates if row["chunk_id"] != rows[0]["chunk_id"]
                ]

            if not candidates:
                return "No other tables share this table's origin document."

            lines = [
                json.dumps(
                    {
                        "source_id": row["source_id"],
                        "chunk_id": row["chunk_id"],
                        "description": row["text"],
                        "schema": format_schema(
                            (row.get("metadata") or {}).get("schema") or []
                        ),
                    },
                    default=str,
                )
                for row in candidates
            ]
            return (
                "Other tables from the same origin document (possible "
                "relationships; load them in sql_query_table via "
                "table_relationships):\n" + "\n".join(lines)
            )

        return execute
