from abc import ABC, abstractmethod
from app.models.document import Document


class MetadataStorage(ABC):
    """Interface for document metadata kept separately from file contents."""

    @abstractmethod
    async def save(self, document: Document) -> None:
        """Saves a document's metadata."""

    @abstractmethod
    async def get(self, document_id: str) -> Document:
        """Retrieves a document's metadata."""

    @abstractmethod
    async def list(self) -> list[Document]:
        """Lists all stored documents' metadata."""

    @abstractmethod
    async def delete(self, document_id: str) -> bool:
        """Deletes a document's metadata."""
