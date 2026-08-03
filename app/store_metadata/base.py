from abc import ABC, abstractmethod
from app.models.document import Document


class MetadataStorage(ABC):
    @abstractmethod
    async def save(self, document: Document) -> None: ...

    @abstractmethod
    async def get(self, document_id: str) -> Document: ...

    @abstractmethod
    async def list(self) -> list[Document]: ...

    @abstractmethod
    async def delete(self, document_id: str) -> bool: ...
