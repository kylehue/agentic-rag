import json

from app.agent_tools.base import AgentTool
from app.agent_tools.common import (
    SOURCE_ID_PARAMETER,
    object_schema,
    resolve_table_document,
    table_chunks_for_source,
    table_schema_entries,
)


class ValidateChunkAsStructuredDataTool(AgentTool):
    """Report whether a source is tabular data and, if so, its table schema(s)."""

    @property
    def name(self) -> str:
        return "validate_chunk_as_structured_data"

    @property
    def description(self) -> str:
        return (
            "Check whether a source is tabular data (a csv or spreadsheet). If "
            "it is, return its table schema(s) as an array: one entry per "
            "table, so a csv returns one and a multi-sheet workbook returns one "
            "per sheet. Each entry has table_name (the name to use for that "
            "table in SQL, which is distinct from the source_id you passed in) "
            "and its schema (columns and types)."
        )

    @property
    def parameters(self) -> dict:
        return object_schema(
            {
                "source_id": {
                    "type": "string",
                    "description": SOURCE_ID_PARAMETER,
                },
            },
            ["source_id"],
        )

    def create_executor(self, rag_service, context):
        sql_storage = rag_service.sql_storage

        async def execute(arguments: dict) -> str:
            source_id = str(arguments.get("source_id", ""))
            await resolve_table_document(sql_storage, source_id, context.chat_id)
            rows = await table_chunks_for_source(sql_storage, source_id, context.chat_id)
            if not rows:
                raise ValueError(f"Source '{source_id}' stores no table.")
            return json.dumps(table_schema_entries(rows), default=str, indent=2)

        return execute
