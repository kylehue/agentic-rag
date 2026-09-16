from abc import ABC, abstractmethod
from typing import IO


class FileStorage(ABC):
    """Interface for storing the original uploaded file."""

    @abstractmethod
    async def upload(
        self,
        *,
        file: IO[bytes],
        file_filename: str,
        file_content_type: str,
        file_dir: str = "",
    ) -> str:
        """Save an uploaded file and return its full path."""

    @abstractmethod
    async def delete(self, full_path: str) -> bool:
        """Deletes the stored file for one document."""

    @abstractmethod
    async def read_bytes(self, full_path: str) -> bytes:
        """Read the stored file."""

    @abstractmethod
    def create_link(self, full_path: str) -> str:
        """A URL path at which this stored file can be retrieved.

        Pure (no I/O): it maps a stored path to a link the app serves.
        """
