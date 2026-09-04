from fastapi import UploadFile

from app.llm.base import LLMProvider
from app.models.chunk import IngestedChunk, RetrievedChunk
from app.models.rag import RagAnswer
from app.services.ingestion import IngestionService
from app.services.retrieval import RetrievalService
from app.utils.string import render_template

ANSWER_PROMPT_TEMPLATE = """Answer the question using only the retrieved evidence.
If the evidence is insufficient, say so. Cite factual claims using the source
filename and evidence id in square brackets, for example [report.pdf:abc:0].

Question:
{query}

Evidence:
{evidence}
"""


class RagService:
    """Facade for the RAG API: ingest, retrieve, answer."""

    def __init__(
        self,
        *,
        llm: LLMProvider,
        ingestion_service: IngestionService,
        retrieval_service: RetrievalService,
    ) -> None:
        self._llm = llm
        self._ingestion_service = ingestion_service
        self._retrieval_service = retrieval_service

    async def ingest(self, file: UploadFile) -> list[IngestedChunk]:
        """Store and index a file."""
        return await self._ingestion_service.ingest(file)

    async def retrieve(self, user_query: str) -> list[RetrievedChunk]:
        """Return evidence without asking the final-answer LLM."""
        return await self._retrieval_service.retrieve(user_query)

    async def answer(
        self,
        user_query: str,
    ) -> RagAnswer:
        """Build a grounded prompt and answer it."""

        chunks = await self.retrieve(user_query)

        evidence = (
            "\n\n".join(
                f"Source: {chunk.source_id}:{chunk.chunk_id}\n{chunk.text}"
                for chunk in chunks
            )
            or "(no evidence retrieved)"
        )

        prompt = render_template(
            ANSWER_PROMPT_TEMPLATE,
            {
                "query": user_query,
                "evidence": evidence,
            },
        )

        answer = await self._llm.answer(prompt)

        return RagAnswer(
            query=user_query,
            answer=answer,
            chunks=chunks,
        )
