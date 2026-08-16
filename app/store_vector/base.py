from abc import ABC, abstractmethod
from typing import Sequence

from app.models.document import DocumentChunk
from app.models.rag import RetrievalCandidate


class VectorStorage(ABC):
    @abstractmethod
    async def add_documents(
        self,
        documents: Sequence[DocumentChunk],
        embeddings: Sequence[Sequence[float]],
    ) -> None:
        """Store chunks and their precomputed embeddings.

        Embeddings are intentionally supplied by the caller so this storage layer
        remains independent of any embedding provider.
        """

    @abstractmethod
    async def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
    ) -> list[RetrievalCandidate]:
        """Return provider-neutral ranked candidates.

        Storage adapters own distance semantics and may omit a normalized score when
        their backend cannot provide a meaningful comparable value.
        """

    @abstractmethod
    async def delete_document(self, document_id: str) -> None:
        """Deletes all vectors belonging to a document."""
