import asyncio
from collections.abc import Sequence
from dataclasses import replace

from app.models.document import RetrievedDocumentChunk
from app.retrievers.base import Retriever
from app.utils.ranking import reciprocal_rank_fusion


class HybridRetriever(Retriever):
    def __init__(
        self,
        retrievers: Sequence[Retriever],
        top_k: int,
    ):
        self._retrievers = retrievers
        self._top_k = top_k

    async def retrieve(
        self,
        user_query: str,
    ) -> list[RetrievedDocumentChunk]:
        tasks = [retriever.retrieve(user_query) for retriever in self._retrievers]

        results = await asyncio.gather(*tasks)

        fused = reciprocal_rank_fusion(
            results,
            id_fn=lambda chunk: chunk.chunk_id,
        )

        return [replace(chunk, score=score) for chunk, score in fused[: self._top_k]]
