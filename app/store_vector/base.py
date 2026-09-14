from abc import ABC, abstractmethod
from typing import Any, Sequence

from app.models.vector import VectorSearchResult


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
        metadatas: Sequence[dict[str, Any]] | None = None,
    ) -> None:
        """Store IDs and their precomputed embeddings.

        Embeddings are intentionally supplied by the caller so this storage
        layer remains independent of embedding providers. `metadatas`, when
        given, must align with `ids` and makes the vectors filterable at
        search time (for example by chat).
        """

    @abstractmethod
    async def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        where: dict[str, Any] | None = None,
    ) -> list[VectorSearchResult]:
        """
        Returns results in descending order (best to worst).

        `where` is an equality filter on the stored metadata (for example
        `{"chat_id": "..."}`); None searches everything.

        Note: Higher score is better.
        """

    @abstractmethod
    async def delete(self, ids: Sequence[str]) -> None:
        """Delete vectors for the supplied chunk IDs."""
