from pathlib import Path
from uuid import uuid4
import shutil

from app.errors.document import InvalidDocumentError
from app.models.document import Document
from app.store_file.base import FileStorage
from app.dependencies import settings
from app.utils.file_type import detect_document_category

UPLOAD_DIR = Path(settings.FILE_LOCAL_STORAGE_DIR)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


class LocalFileStorage(FileStorage):
    async def save(self, file):
        document_id = str(uuid4())

        if not file.filename:
            raise InvalidDocumentError()

        extension = Path(file.filename).suffix

        path = UPLOAD_DIR / f"{document_id}{extension}"

        with path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        return Document(
            id=document_id,
            filename=file.filename,
            extension=extension,
            path=str(path),
            category=detect_document_category(extension),
        )

    async def delete(self, document_id):
        matches = list(UPLOAD_DIR.glob(f"{document_id}.*"))

        if not matches:
            return False

        matches[0].unlink()

        return True
