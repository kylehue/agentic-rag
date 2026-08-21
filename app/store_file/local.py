from pathlib import Path
from shutil import copyfileobj
from typing import IO
from uuid import uuid4

from app.store_file.base import FileStorage


class LocalFileStorage(FileStorage):
    """Stores files in a local directory."""

    def __init__(
        self,
        storage_dir: str | Path,
    ):
        self._storage_dir = Path(storage_dir)
        self._storage_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    async def upload(
        self,
        *,
        file: IO[bytes],
        file_filename: str,
        file_content_type: str,
        file_dir: str = "",
    ) -> str:
        # Preserve the original extension.
        extension = Path(file_filename).suffix

        # Generate a unique filename.
        filename = f"{uuid4()}{extension}"

        # Resolve the destination directory.
        directory = self._storage_dir / file_dir
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        # Full destination path.
        path = directory / filename

        # Write the file.
        with path.open("wb") as output:
            copyfileobj(file, output)

        return str(path)

    async def delete(
        self,
        full_path: str,
    ) -> bool:
        path = Path(full_path)

        if not path.is_file():
            return False

        path.unlink()

        return True

    async def read_bytes(
        self,
        full_path: str,
    ) -> bytes:
        path = Path(full_path)

        if not path.is_file():
            raise FileNotFoundError(f"File not found: {full_path}")

        return path.read_bytes()
