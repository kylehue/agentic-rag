import json
from collections import defaultdict
from collections.abc import Sequence

from app.llm.base import LLMProvider
from app.models.document import DocumentCategory
from app.models.rag import RetrievalCandidate, RetrievedEvidence, RetrievalRequest
from app.retrievers.base import Retriever
from app.store_sql2.base import SqlStorage

SQL_PROMPT = """You generate exactly one read-only {dialect} SQL query
for a retrieval system.

Return SQL only. Do not use Markdown, code fences, explanations, or comments.

Return an empty response if the retrieved spreadsheet context does not contain
enough information to answer the question reliably with SQL.

Use only the tables and columns provided in the schema.
Never invent tables, columns, values, or relationships.

ALL COLUMNS ARE NULLABLE.

Generate valid {dialect} SQL and account for NULL values correctly:
- Never use = NULL or != NULL.
- Use IS NULL / IS NOT NULL for NULL checks.
- Do not assume nullable columns contain values.
- Use COALESCE only when a NULL replacement is semantically appropriate.
- Preserve the meaning of missing data rather than silently converting it.
- Avoid unsafe or destructive statements.
- Generate exactly one read-only query.

Question:
{query}

Retrieved spreadsheet context and SQL schema:
{context}
"""

SPREADSHEET_DB = "spreadsheets"


class SpreadsheetRetriever(Retriever):
    """Uses retrieved sheet context to optionally query a spreadsheet's SQL tables.

    Flow: retrieve() > _retrieve_document() > SQL query > RetrievedEvidence
    """

    name = "spreadsheet"

    def __init__(self, llm: LLMProvider, sql_storage: SqlStorage):
        """Receive the LLM that writes SQL and the storage that safely runs it."""
        self._llm = llm
        self._sql_storage = sql_storage

    async def retrieve(
        self, request: RetrievalRequest, candidates: Sequence[RetrievalCandidate]
    ) -> list[RetrievedEvidence]:
        """Returns spreadsheet candidates with their SQL results attached for the LLM."""
        groups: dict[str, list[RetrievalCandidate]] = defaultdict(list)
        for item in candidates:
            if item.document.category is DocumentCategory.SPREADSHEET:
                groups[item.document.id].append(item)

        evidence: list[RetrievedEvidence] = []
        for document_candidates in groups.values():
            evidence.extend(await self._retrieve_document(request, document_candidates))
        return evidence

    async def _retrieve_document(
        self, request: RetrievalRequest, candidates: list[RetrievalCandidate]
    ) -> list[RetrievedEvidence]:
        """Ask for SQL when possible, otherwise return the retrieved sheet context."""
        document = candidates[0].document
        if not any(item.metadata.get("sql_table") for item in candidates):
            return [self._chunk_evidence(item) for item in candidates]
        context = "\n\n".join(self._context(item) for item in candidates)
        try:
            sql = self._clean_sql(
                await self._llm.answer(
                    SQL_PROMPT.format(query=request.query, context=context)
                )
            )
        except Exception:
            return [self._chunk_evidence(item) for item in candidates]
        if not sql:
            return [self._chunk_evidence(item) for item in candidates]

        try:
            result = await self._sql_storage.query(
                db_name=SPREADSHEET_DB,
                sql_query=sql,
                limit=request.max_sql_rows,
            )
        except Exception as error:
            # Keep the retrieved source context useful when generated SQL is invalid.
            return [
                *[self._chunk_evidence(item) for item in candidates],
                RetrievedEvidence(
                    id=f"{document.id}:sql-error",
                    retriever=self.name,
                    document=document,
                    content="The spreadsheet SQL query could not be run; use the retrieved sheet context instead.",
                    metadata={"sql": sql, "error": str(error)},
                ),
            ]

        return [
            RetrievedEvidence(
                id=f"{document.id}:sql-result",
                retriever=self.name,
                document=document,
                content=self._result_content(
                    sql,
                    result.columns,
                    result.rows,
                    self._workbook_context(candidates),
                ),
                metadata={"sql": sql, "columns": result.columns, "rows": result.rows},
            )
        ]

    @staticmethod
    def _context(candidate: RetrievalCandidate) -> str:
        """Add the SQL table mapping to a candidate before giving it to the LLM."""
        schema = {
            "sql_table": candidate.metadata.get("sql_table"),
            "sql_columns": candidate.metadata.get("sql_columns", []),
        }
        return f"{candidate.text}\nSQL mapping: {json.dumps(schema, default=str)}"

    def _chunk_evidence(self, candidate: RetrievalCandidate) -> RetrievedEvidence:
        """Turn one retrieved spreadsheet chunk into fallback text evidence."""
        return RetrievedEvidence(
            id=candidate.id,
            retriever=self.name,
            document=candidate.document,
            content=candidate.text,
            metadata=candidate.metadata,
            score=candidate.score,
        )

    @staticmethod
    def _clean_sql(response: str) -> str:
        """Remove an optional Markdown code fence from LLM-produced SQL."""
        value = response.strip()
        if value.startswith("```"):
            value = value.split("\n", 1)[-1]
            if value.rstrip().endswith("```"):
                value = value.rstrip()[:-3].rstrip()
        return value

    @staticmethod
    def _workbook_context(candidates: Sequence[RetrievalCandidate]) -> str:
        """Collect unique workbook descriptions for the final SQL evidence."""
        descriptions = {
            str(item.metadata["workbook_description"])
            for item in candidates
            if item.metadata.get("workbook_description")
        }
        return "\n".join(sorted(descriptions))

    @staticmethod
    def _result_content(
        sql: str,
        columns: list[str],
        rows: list[tuple[object, ...]],
        workbook_context: str,
    ) -> str:
        """Format SQL, rows, and workbook context as readable final evidence."""
        content = (
            "Spreadsheet SQL query:\n"
            + sql
            + "\nResults:\n"
            + json.dumps([dict(zip(columns, row)) for row in rows], default=str)
        )
        return (
            f"Workbook description:\n{workbook_context}\n\n{content}"
            if workbook_context
            else content
        )
