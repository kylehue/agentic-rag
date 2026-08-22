from abc import ABC, abstractmethod
from typing import Sequence


class VectorStorage(ABC):
    @abstractmethod
    async def close(self) -> None:
        """Release resources held by the vector store."""
        pass

    @abstractmethod
    async def add(
        self,
        ids: Sequence[str],
        embeddings: Sequence[Sequence[float]],
    ) -> None:
        """Store IDs and their precomputed embeddings.

        Embeddings are intentionally supplied by the caller so this storage layer
        remains independent of embedding providers and chunk metadata.
        """

    @abstractmethod
    async def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
    ) -> list[str]:
        """Return nearest chunk IDs in ranked order."""

    @abstractmethod
    async def delete(self, ids: Sequence[str]) -> None:
        """Delete vectors for the supplied chunk IDs."""
