import json
from dataclasses import replace

from app.finalizers.base import Finalizer
from app.llm.base import LLMProvider
from app.models.chunk import ChunkCategory, RetrievedChunk
from app.store_sql.base import SqlStorage
from app.utils.string import render_template

SQL_PROMPT_TEMPLATE = """You generate at most one read-only {dialect} SQL query
for a retrieval system.

Return SQL only. Do not use Markdown, code fences, explanations, or comments.

IMPORTANT:
Only generate SQL when the user's question actually requires querying the
underlying spreadsheet rows.

Return an empty response if the question can be answered from the retrieved
spreadsheet context alone without executing SQL.

Return an empty response if the supplied spreadsheet context does not contain
enough information to answer the question reliably.

The retrieved context may describe multiple related spreadsheet tables.
You may JOIN multiple tables when the question requires information from more
than one table.

You may ONLY use tables and columns explicitly provided in the retrieved
spreadsheet context.

Never invent:
- tables
- columns
- relationships
- values
- join conditions

ALL TABLE COLUMNS ARE NULLABLE.

Generate valid {dialect} SQL and account for NULL values correctly:
- Never use = NULL or != NULL.
- Use IS NULL / IS NOT NULL for NULL checks.
- Do not assume nullable columns contain values.
- Use COALESCE only when a NULL replacement is semantically appropriate.
- Preserve the meaning of missing data rather than silently converting it.
- Use NULL-safe expressions for comparisons and calculations.
- Do not generate INSERT, UPDATE, DELETE, DROP, ALTER, CREATE, PRAGMA,
  ATTACH, or any other mutating statement.
- Generate exactly one read-only SELECT statement.
- Only reference tables and columns supplied in the context.
- Only use relationships explicitly supported by the context.

Question:
{query}

Retrieved spreadsheet context:
{context}
"""


class SpreadsheetFinalizer(Finalizer):
    def __init__(
        self,
        llm: LLMProvider,
        sql_storage: SqlStorage,
        *,
        max_sql_rows: int | None = 50,
    ):
        self._llm = llm
        self._sql_storage = sql_storage
        self._max_sql_rows = max_sql_rows

    async def finalize(
        self,
        user_query: str,
        chunk: RetrievedChunk,
    ) -> RetrievedChunk:
        """Optionally query the underlying spreadsheet and append SQL evidence."""

        if chunk.category is not ChunkCategory.SPREADSHEET:
            return chunk

        try:
            prompt = render_template(
                SQL_PROMPT_TEMPLATE,
                {
                    "dialect": self._sql_storage.get_sql_dialect(),
                    "query": user_query,
                    "context": chunk.text,
                },
            )

            response = await self._llm.answer(prompt)
        except Exception:
            # SQL generation is optional. Keep the original evidence usable.
            return chunk

        sql = self._clean_sql(response)

        # Empty response means the LLM decided that SQL is unnecessary.
        if not sql:
            return chunk

        try:
            results = await self._sql_storage.query(
                sql,
                limit=self._max_sql_rows,
            )
        except Exception:
            # Keep the original context if generated SQL cannot be executed.
            return chunk

        return replace(
            chunk,
            text=self._append_sql_evidence(
                chunk.text,
                sql,
                results,
            ),
        )

    @staticmethod
    def _clean_sql(response: str) -> str:
        """Remove optional Markdown fences from LLM-generated SQL."""

        value = response.strip()

        if not value:
            return ""

        if value.startswith("```"):
            lines = value.splitlines()

            if lines:
                lines = lines[1:]

            if lines and lines[-1].strip() == "```":
                lines.pop()

            value = "\n".join(lines).strip()

        return value

    @staticmethod
    def _append_sql_evidence(
        original_text: str,
        sql: str,
        results: object,
    ) -> str:
        """Append the generated query and its SQL results to chunk text."""

        result_text = json.dumps(
            results,
            ensure_ascii=False,
            default=str,
        )

        return (
            f"{original_text}\n\n"
            "Spreadsheet SQL query:\n"
            f"{sql}\n\n"
            "Spreadsheet SQL results:\n"
            f"{result_text}"
        )
