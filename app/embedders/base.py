from abc import ABC, abstractmethod
from collections.abc import Sequence


class Embedder(ABC):
    @abstractmethod
    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed document chunks in the same order as the supplied texts."""

    @abstractmethod
    async def embed_query(self, text: str) -> list[float]:
        """Embed a retrieval query."""
