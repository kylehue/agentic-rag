from abc import ABC, abstractmethod


class MetadataStorage(ABC):
    """Interface for JSON records kept separately from their primary data."""

    @abstractmethod
    async def upsert(self, collection_name: str, id: str, document_json: str) -> None:
        """Add or replace a JSON record in a collection."""

    @abstractmethod
    async def get(self, id: str) -> str:
        """Retrieve a JSON record by ID.

        Raises:
            KeyError: If no record exists with this ID.
        """

    @abstractmethod
    async def list(self) -> list[str]:
        """List every stored JSON record."""

    @abstractmethod
    async def delete(self, id: str) -> bool:
        """Delete a JSON record and report whether it existed."""
