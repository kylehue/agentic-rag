import json
import re

from app.agent_tools.base import AgentTool
from app.agent_tools.common import object_schema

# Chat ids are system-generated (uuid4 hex); this guard keeps the value that
# gets inlined into the wrapped query free of SQL metacharacters.
_CHAT_ID_SHAPE = re.compile(r"[\w-]+")

# The query's row cap lives in the SQL itself, so a LIMIT clause is required
# and enforced here rather than applied on the tool side.
_HAS_LIMIT = re.compile(r"\blimit\b", re.IGNORECASE)


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
            "the current chat, and a query without it is rejected. The query "
            "must end with a LIMIT clause; pick the limit for the question, not "
            "for the table: only as many rows as the answer needs."
        )

    @property
    def parameters(self) -> dict:
        return object_schema(
            {
                "sql": {
                    "type": "string",
                    "description": (
                        "A read-only SQL SELECT query, with chat_id in the "
                        "SELECT list and a LIMIT clause. Choose the limit "
                        "smartly: the fewest rows that answer the question "
                        "(a COUNT needs 1). Extra rows just crowd the context."
                    ),
                },
            },
            ["sql"],
        )

    def create_executor(self, rag_service, context):
        sql_storage = rag_service.sql_storage

        async def execute(arguments: dict) -> str:
            sql = str(arguments.get("sql", "")).strip()
            first_word = sql.split(None, 1)[0].upper() if sql else ""
            # SELECT only; no WITH, because a CTE can wrap a DELETE in
            # SQLite; the storage layer enforces the same rule as the
            # backstop.
            if first_word != "SELECT":
                return "Error: only read-only SELECT queries are allowed."

            if not _HAS_LIMIT.search(sql):
                return (
                    "Error: include a LIMIT clause in the query, sized to how "
                    "many rows the question actually needs."
                )

            if context.chat_id is not None:
                if not _CHAT_ID_SHAPE.fullmatch(context.chat_id):
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
                    f"WHERE scoped.chat_id = '{context.chat_id}'"
                )

            # No tool-side cap: the LIMIT clause in the SQL is the row bound.
            rows = await sql_storage.query(sql, None)
            if not rows:
                return "No rows."

            lines = []
            for row in rows:
                # A chunk-shaped row (chunk_id + origin_source_id) is citable
                # evidence; register it so the answer can cite it by number.
                origin = row.get("origin_source_id")
                chunk_id = row.get("chunk_id")
                body = json.dumps(row, default=str)
                if origin and chunk_id:
                    index = context.evidence.register(origin, chunk_id)
                    lines.append(f"[{index}] {body}")
                else:
                    lines.append(body)
            return "Rows (one per line):\n" + "\n".join(lines)

        return execute
