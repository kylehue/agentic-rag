import json
import re

from app.agent_tools.base import AgentTool
from app.agent_tools.common import SQL_MAX_ROWS, object_schema

# Chat ids are system-generated (uuid4 hex); this guard keeps the value that
# gets inlined into the wrapped query free of SQL metacharacters.
_CHAT_ID_SHAPE = re.compile(r"[\w-]+")


class PerformSqlToDocumentRecordsTool(AgentTool):
    """Query the chunk/document metadata records (read-only SELECT)."""

    @property
    def name(self) -> str:
        return "perform_sql_to_document_records"

    @property
    def description(self) -> str:
        return (
            "Run a read-only SELECT query against the RAG document database "
            "(the chunk and document records described in the Records section). "
            "Always include chat_id in the SELECT list; results are bounded to "
            "the current chat, and a query without it is rejected."
        )

    @property
    def parameters(self) -> dict:
        return object_schema(
            {
                "sql": {
                    "type": "string",
                    "description": "A read-only SQL SELECT query.",
                },
                "limit": {
                    "type": "integer",
                    "description": (
                        f"Maximum rows to return (default {SQL_MAX_ROWS})."
                    ),
                },
            },
            ["sql"],
        )

    def create_executor(self, rag_service, chat_id=None):
        sql_storage = rag_service.sql_storage

        async def execute(arguments: dict) -> str:
            sql = str(arguments.get("sql", "")).strip()
            first_word = sql.split(None, 1)[0].upper() if sql else ""
            # SELECT only; no WITH, because a CTE can wrap a DELETE in
            # SQLite; the storage layer enforces the same rule as the
            # backstop.
            if first_word != "SELECT":
                return "Error: only read-only SELECT queries are allowed."

            try:
                limit = int(arguments.get("limit") or SQL_MAX_ROWS)
            except (TypeError, ValueError):
                limit = SQL_MAX_ROWS
            limit = max(1, min(limit, SQL_MAX_ROWS))

            if chat_id is not None:
                if not _CHAT_ID_SHAPE.fullmatch(chat_id):
                    return "Error: invalid chat scope."
                # Results are bounded to this chat by wrapping the query and
                # filtering on chat_id, which the query must therefore
                # select. Check for it upfront and say so plainly, rather than
                # letting it fail with a cryptic SQL error.
                if "chat_id" not in sql.lower():
                    return (
                        "Error: the query must include chat_id in its SELECT "
                        "list so results are bounded to the current chat."
                    )
                sql = (
                    f"SELECT * FROM ({sql}) AS scoped "
                    f"WHERE scoped.chat_id = '{chat_id}'"
                )

            rows = await sql_storage.query(sql, limit)
            if not rows:
                return "No rows."
            return "Rows (JSON, one per line):\n" + "\n".join(
                json.dumps(row, default=str) for row in rows
            )

        return execute
