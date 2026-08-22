from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class VectorSearchResult:
    id: str
    score: float


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
    ) -> list[VectorSearchResult]:
        """
        Returns results in descending order (best to worst).

        Note: Higher score is better.
        """

    @abstractmethod
    async def delete(self, ids: Sequence[str]) -> None:
        """Delete vectors for the supplied chunk IDs."""
