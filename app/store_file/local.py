from pathlib import Path
from uuid import uuid4
import shutil

from fastapi import UploadFile

from app.models.document import Document
from app.store_file.base import FileStorage
from app.dependencies import settings

UPLOAD_DIR = Path(settings.FILE_LOCAL_STORAGE_DIR)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


class LocalFileStorage(FileStorage):
    async def save(self, file: UploadFile) -> Document:
        document_id = str(uuid4())

        extension = Path(file.filename).suffix

        path = UPLOAD_DIR / f"{document_id}{extension}"

        with path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        return Document(
            id=document_id,
            filename=file.filename,
            path=str(path),
        )

    async def delete(self, document_id: str) -> bool:
        matches = list(UPLOAD_DIR.glob(f"{document_id}.*"))

        if not matches:
            return False

        matches[0].unlink()

        return True
