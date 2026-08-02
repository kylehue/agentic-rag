from abc import ABC, abstractmethod
from typing import Any


class VectorStorage(ABC):
    @abstractmethod
    def add_documents(self, documents: list[Any]) -> None:
        """Store documents in the vector database."""
        pass

    @abstractmethod
    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
    ) -> list[Any]:
        """Return the most relevant documents."""
        pass

    @abstractmethod
    def delete_document(self, document_id: str) -> None:
        """Deletes all vectors belonging to a document."""
        pass
