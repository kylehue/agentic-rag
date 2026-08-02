from abc import ABC, abstractmethod
from fastapi import UploadFile
from app.models.document import Document


class FileStorage(ABC):
    @abstractmethod
    async def save(self, file: UploadFile) -> Document:
        """Save an uploaded file."""

    @abstractmethod
    async def delete(self, document_id: str) -> bool:
        """Deletes a file."""
