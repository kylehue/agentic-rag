import json

from app.agent.tools import AgentTool
from app.agent_tools.common import object_schema
from app.core.config import settings
from app.store_sql.base import SqlStorage

# Row cap keeps tool output small enough for the model's context.
QUERY_CHUNKS_MAX_ROWS = 20


def make_query_chunks_tool(sql_storage: SqlStorage) -> AgentTool:
    """Query the chunk/document metadata database (read-only SELECT)."""

    async def execute(arguments: dict) -> str:
        sql = str(arguments.get("sql", "")).strip()
        first_word = sql.split(None, 1)[0].upper() if sql else ""
        # SELECT only — no WITH, because a CTE can wrap a DELETE in SQLite;
        # the storage layer enforces the same rule as the backstop.
        if first_word != "SELECT":
            return "Error: only read-only SELECT queries are allowed."

        try:
            limit = int(arguments.get("limit") or QUERY_CHUNKS_MAX_ROWS)
        except (TypeError, ValueError):
            limit = QUERY_CHUNKS_MAX_ROWS
        limit = max(1, min(limit, QUERY_CHUNKS_MAX_ROWS))

        rows = await sql_storage.query(sql, limit)
        if not rows:
            return "No rows."
        return (
            "Rows (JSON, one per line):\n"
            + "\n".join(json.dumps(row, default=str) for row in rows)
        )

    return AgentTool(
        name="query_chunks",
        description=(
            "Run a read-only SELECT query against the document database. "
            f"Tables: {settings.CHUNK_TABLE_NAME} (columns: id, chunk_id, "
            "source_id, parent_source_id, origin_source_id, plugin, text, "
            f"metadata) and {settings.DOCUMENT_METADATA_TABLE_NAME} (columns: "
            "id, source_id, file_path, file_content_type, file_filename, "
            "file_orig_filename, is_origin)."
        ),
        parameters=object_schema(
            {
                "sql": {
                    "type": "string",
                    "description": "A read-only SQL SELECT query.",
                },
                "limit": {
                    "type": "integer",
                    "description": (
                        f"Maximum rows to return (default {QUERY_CHUNKS_MAX_ROWS})."
                    ),
                },
            },
            ["sql"],
        ),
        execute=execute,
    )
