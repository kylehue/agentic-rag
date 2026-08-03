import json
from pathlib import Path

from app.errors.document import DocumentNotFoundError
from app.models.document import Document
from app.store_metadata.base import MetadataStorage
from app.dependencies import settings

METADATA_DIR = Path(settings.METADATA_LOCAL_STORAGE_DIR)
METADATA_DIR.mkdir(parents=True, exist_ok=True)

METADATA_FILE = METADATA_DIR / "documents.json"

if not METADATA_FILE.exists():
    METADATA_FILE.write_text("{}", encoding="utf-8")


class LocalMetadataStorage(MetadataStorage):
    def _load(self) -> dict:
        with METADATA_FILE.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _save(self, data: dict):
        with METADATA_FILE.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)

    async def save(self, document: Document) -> None:
        metadata = self._load()

        metadata[document.id] = {
            "id": document.id,
            "filename": document.filename,
            "path": str(document.path),
        }

        self._save(metadata)

    async def get(self, document_id: str) -> Document:
        metadata = self._load()

        item = metadata.get(document_id)

        if item is None:
            raise DocumentNotFoundError(document_id)

        return Document(
            id=item["id"],
            filename=item["filename"],
            path=item["path"],
        )

    async def list(self) -> list[Document]:
        metadata = self._load()

        return [
            Document(
                id=item["id"],
                filename=item["filename"],
                path=item["path"],
            )
            for item in metadata.values()
        ]

    async def delete(self, document_id: str) -> bool:
        metadata = self._load()

        if document_id not in metadata:
            return False

        del metadata[document_id]

        self._save(metadata)

        return True
