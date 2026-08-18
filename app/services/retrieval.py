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
from app.store_metadata.base import MetadataStorage
from app.utils.chunks import deserialize_chunk


class RetrievalService:
    """Finds relevant chunks and lets each retriever turn them into evidence.

    Flow: retrieve() > _search_candidates() > retriever.retrieve()
    """

    def __init__(
        self,
        embedder: Embedder,
        vector_storage: VectorStorage,
        metadata_storage: MetadataStorage,
        retrievers: Sequence[Retriever],
    ):
        self._embedder = embedder
        self._vector_storage = vector_storage
        self._metadata_storage = metadata_storage
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
        chunk_ids = await self._vector_storage.search(query_embedding, request.top_k)
        records = await asyncio.gather(
            *(self._metadata_storage.get(chunk_id) for chunk_id in chunk_ids),
            return_exceptions=True,
        )
        candidates = []
        for chunk_id, record in zip(chunk_ids, records):
            try:
                if isinstance(record, Exception):
                    continue
                chunk = deserialize_chunk(record)  # type: ignore
            except ValueError:
                continue
            candidates.append(
                RetrievalCandidate(
                    id=chunk.id,
                    text=chunk.text,
                    document=chunk.document,
                    metadata=chunk.metadata,
                    binary_content=chunk.binary_content,
                    binary_mime_type=chunk.binary_mime_type,
                )
            )
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
