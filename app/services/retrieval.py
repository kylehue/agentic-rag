import asyncio
from collections.abc import Iterable, Sequence

from app.embedders.base import Embedder
from app.models.rag import (
    RetrievalCandidate,
    RetrievalResult,
    RetrievalRequest,
    RetrievedEvidence,
)
from app.retrievers.base import Retriever
from app.store_vector.base import VectorStorage


class RetrievalService:
    """Finds relevant chunks and lets each retriever turn them into evidence.

    Flow: retrieve() > _search_candidates() > retriever.retrieve()
    """

    def __init__(
        self,
        embedder: Embedder,
        vector_storage: VectorStorage,
        retrievers: Sequence[Retriever],
    ):
        self._embedder = embedder
        self._vector_storage = vector_storage
        self._retrievers = tuple(retrievers)
        names = [retriever.name for retriever in retrievers]
        if len(names) != len(set(names)):
            raise ValueError("Retriever names must be unique")

    async def retrieve(self, request: RetrievalRequest) -> RetrievalResult:
        """Search once, then run all applicable evidence retrievers in parallel."""
        candidates = await self._search_candidates(request)
        outputs = await asyncio.gather(
            *(retriever.retrieve(request, candidates) for retriever in self._retrievers)
        )
        return RetrievalResult(
            query=request.query,
            evidence=self._deduplicate(item for output in outputs for item in output),
        )

    async def _search_candidates(
        self, request: RetrievalRequest
    ) -> list[RetrievalCandidate]:
        """Embed the question, search vectors, and apply any category filter."""
        query_embedding = await self._embedder.embed_query(request.query)
        candidates = await self._vector_storage.search(query_embedding, request.top_k)
        if request.categories is None:
            return candidates
        return [
            item for item in candidates if item.document.category in request.categories
        ]

    @staticmethod
    def _deduplicate(evidence: Iterable[RetrievedEvidence]) -> list[RetrievedEvidence]:
        """Keep the first copy when two retrievers produce the same evidence."""
        unique: dict[tuple[str, str], RetrievedEvidence] = {}
        for item in evidence:
            unique.setdefault((item.retriever, item.id), item)
        return list(unique.values())
