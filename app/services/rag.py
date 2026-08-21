from fastapi import UploadFile

from app.models.llm import LLMAttachment
from app.llm.base import LLMProvider
from app.models.rag import RagAnswer, RetrievalRequest
from app.services.ingestion import IngestionService
from app.services.retrieval import RetrievalService

ANSWER_PROMPT = """Answer the question using only the retrieved evidence.
If the evidence is insufficient, say so. Cite factual claims using the source
filename and evidence id in square brackets, for example [report.pdf:abc:0].

Question:
{query}

Evidence:
{evidence}
"""


class RagService:
    def __init__(
        self,
        ingestion_service: IngestionService,
        retrieval_service: RetrievalService,
        llm: LLMProvider,
    ):
        self._ingestion_service = ingestion_service
        self._retrieval_service = retrieval_service
        self._llm = llm

    async def ingest(self, file: UploadFile):
        """Store and index a file."""
        return await self._ingestion_service.ingest(file)

    async def retrieve(self, request: RetrievalRequest):
        """Return evidence without asking the final-answer LLM."""
        return await self._retrieval_service.retrieve(request)

    async def answer(self, request: RetrievalRequest) -> RagAnswer:
        """Build a grounded prompt and answer it with text."""
        result = await self.retrieve(request)
        evidence = (
            "\n\n".join(
                f"Source: {item.document.file_filename}:{item.id}\n{item.content}"
                for item in result.evidence
            )
            or "(no evidence retrieved)"
        )
        prompt = ANSWER_PROMPT.format(query=request.query, evidence=evidence)
        attachments = tuple(
            LLMAttachment(
                item.binary_content,
                item.binary_mime_type or "application/octet-stream",
            )
            for item in result.evidence
            if item.binary_content is not None
        )
        answer = await self._llm.answer(prompt, attachments=attachments)
        return RagAnswer(query=request.query, answer=answer, evidence=result.evidence)
