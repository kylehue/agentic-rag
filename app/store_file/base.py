from abc import ABC, abstractmethod
from fastapi import UploadFile
from app.models.document import Document


class FileStorage(ABC):
    """Interface for storing the original uploaded file."""

    @abstractmethod
    async def save(self, file: UploadFile) -> Document:
        """Save an uploaded file and return its document record."""

    @abstractmethod
    async def delete(self, document_id: str) -> bool:
        """Delete the stored file for one document."""
