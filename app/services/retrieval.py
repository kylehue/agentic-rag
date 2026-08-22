import asyncio
from app.finalizers.base import Finalizer
from app.models.chunk import RetrievedChunk
from app.retrievers.base import Retriever


class RetrievalService:
    def __init__(
        self,
        retriever: Retriever,
        finalizer: Finalizer,
    ):
        self._retriever = retriever
        self._finalizer = finalizer

    async def retrieve(self, user_query: str) -> list[RetrievedChunk]:
        """Retrieves chunks given a user query."""

        chunks = await self._retriever.retrieve(user_query)
        tasks = [self._finalizer.finalize(user_query, chunk) for chunk in chunks]
        results = await asyncio.gather(*tasks)

        return results
