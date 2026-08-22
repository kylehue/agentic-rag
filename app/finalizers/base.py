from abc import ABC, abstractmethod

from app.models.chunk import RetrievedChunk


class Finalizer(ABC):
    @abstractmethod
    async def finalize(
        self,
        user_query: str,
        chunk: RetrievedChunk,
    ) -> RetrievedChunk:
        """Finalize the chunk content for the LLM."""
