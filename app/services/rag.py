from collections.abc import Sequence

from app.embedders.base import Embedder
from app.llm.base import LLMProvider
from app.models.chunk import IngestedChunk, RetrievedChunk
from app.models.rag import RagAnswer
from app.plugin.base import Plugin
from app.plugin.context import IngestionFile
from app.plugin.registry import PluginRegistry
from app.retrievers.base import Retriever
from app.services.ingestion import IngestionService
from app.services.retrieval import RetrievalService
from app.store_file.base import FileStorage
from app.store_sql.base import SqlStorage
from app.store_vector.base import VectorStorage
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
    def __init__(
        self,
        *,
        llm: LLMProvider,
        embedder: Embedder,
        retriever: Retriever,
        vector_storage: VectorStorage,
        sql_storage: SqlStorage,
        file_storage: FileStorage,
        plugins: Sequence[Plugin] | None = None,
    ) -> None:
        self._llm = llm

        registry = PluginRegistry()
        for plugin in (plugins if plugins is not None else []):
            registry.register(plugin)

        self._ingestion_service = IngestionService(
            registry=registry,
            llm=llm,
            embedder=embedder,
            vector_storage=vector_storage,
            sql_storage=sql_storage,
            file_storage=file_storage,
        )
        self._retrieval_service = RetrievalService(
            retriever=retriever,
            llm=llm,
            registry=registry,
        )

    async def initialize(self) -> None:
        await self._ingestion_service.initialize()

    async def ingest(
        self,
        file_bytes: bytes,
        filename: str,
        content_type: str,
        description: str | None = None,
    ) -> list[IngestedChunk]:
        """Store and index a file."""
        file = IngestionFile(
            filename=filename,
            content_type=content_type,
            file_bytes=file_bytes,
            description=description,
        )
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
