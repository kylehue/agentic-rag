import json
from collections import defaultdict
from collections.abc import Sequence

from app.llm.base import LLMProvider
from app.models.document import DocumentCategory
from app.models.rag import RetrievalCandidate, RetrievedEvidence, RetrievalRequest
from app.retrievers.base import Retriever
from app.store_sql.base import SqlStorage

SQL_PROMPT = """You create one SQLite read-only query for a retrieval system.
Use only the generated table and column names in the supplied schema. Return SQL
only, with no Markdown. Return an empty response when the retrieved spreadsheet
context cannot answer the question more accurately with a SQL query. Never query
tables not supplied below.

Question:
{query}

Retrieved spreadsheet context and SQL schema:
{context}
"""


class SpreadsheetRetriever(Retriever):
    name = "spreadsheet"

    def __init__(self, llm: LLMProvider, sql_storage: SqlStorage):
        self._llm = llm
        self._sql_storage = sql_storage

    async def retrieve(
        self, request: RetrievalRequest, candidates: Sequence[RetrievalCandidate]
    ) -> list[RetrievedEvidence]:
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
                document.id, sql, max_rows=request.max_sql_rows
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
        schema = {
            "sql_table": candidate.metadata.get("sql_table"),
            "sql_columns": candidate.metadata.get("sql_columns", []),
        }
        return f"{candidate.text}\nSQL mapping: {json.dumps(schema, default=str)}"

    def _chunk_evidence(self, candidate: RetrievalCandidate) -> RetrievedEvidence:
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
        value = response.strip()
        if value.startswith("```"):
            value = value.split("\n", 1)[-1]
            if value.rstrip().endswith("```"):
                value = value.rstrip()[:-3].rstrip()
        return value

    @staticmethod
    def _workbook_context(candidates: Sequence[RetrievalCandidate]) -> str:
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
