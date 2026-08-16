import json
from pathlib import Path

from app.errors.document import DocumentNotFoundError
from app.models.document import Document
from app.store_metadata.base import MetadataStorage
from app.core.config import settings

METADATA_DIR = Path(settings.METADATA_LOCAL_STORAGE_DIR)
METADATA_DIR.mkdir(parents=True, exist_ok=True)

METADATA_FILE = METADATA_DIR / "documents.json"

if not METADATA_FILE.exists():
    METADATA_FILE.write_text("{}", encoding="utf-8")


class LocalMetadataStorage(MetadataStorage):
    """Keeps document records in one local JSON file."""

    def _load(self) -> dict:
        """Read the full metadata JSON file into memory."""
        with METADATA_FILE.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _save(self, data: dict):
        """Write the full metadata dictionary back to the JSON file."""
        with METADATA_FILE.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)

    async def save(self, document):
        """Add or replace one document record."""
        metadata = self._load()

        metadata[document.id] = document.model_dump()

        self._save(metadata)

    async def get(self, document_id):
        """Return one document record or raise when it does not exist."""
        metadata = self._load()

        item = metadata.get(document_id)

        if item is None:
            raise DocumentNotFoundError(document_id)

        return Document.model_validate(item)

    async def list(self):
        """Return every document record."""
        metadata = self._load()

        return [Document.model_validate(item) for item in metadata.values()]

    async def delete(self, document_id):
        """Remove one document record and report whether it existed."""
        metadata = self._load()

        if document_id not in metadata:
            return False

        del metadata[document_id]

        self._save(metadata)

        return True
