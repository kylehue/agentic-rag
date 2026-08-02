from abc import ABC, abstractmethod
from fastapi import UploadFile
from app.models.document import Document


class MetadataStorage(ABC):
    @abstractmethod
    async def save(self, file: UploadFile) -> Document:
        """Save an uploaded file."""

    @abstractmethod
    async def get(self, document_id: str) -> Document | None:
        """Get a document by ID."""

    @abstractmethod
    async def list(self) -> list[Document]:
        """List all stored documents."""

    @abstractmethod
    async def delete(self, document_id: str) -> bool:
        """Delete a document."""
