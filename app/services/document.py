from fastapi import UploadFile

from app.errors.document import DocumentNotFoundError
from app.models.document import Document
from app.store_file.base import FileStorage
from app.store_metadata.base import MetadataStorage
from app.store_sql.base import SqlStorage

DOCUMENT_METADATA_COLLECTION = "documents"


class DocumentService:
    """Coordinates file, metadata, and spreadsheet-table lifecycle operations.

    Flow: upload() > FileStorage.save() > MetadataStorage.save()
    """

    def __init__(
        self,
        file_storage: FileStorage,
        metadata_storage: MetadataStorage,
        sql_storage: SqlStorage | None = None,
    ):
        self.file_storage = file_storage
        self.metadata_storage = metadata_storage
        self.sql_storage = sql_storage

    async def upload(self, file: UploadFile) -> Document:
        """Save an upload and register the resulting document metadata."""
        # Save the file first because metadata needs its generated document details.
        document = await self.file_storage.save(file)

        # Serialization belongs to this caller; the metadata store only persists JSON.
        await self.metadata_storage.upsert(
            DOCUMENT_METADATA_COLLECTION, document.id, document.model_dump_json()
        )

        return document

    async def list(self) -> list[Document]:
        """List every saved document record."""
        records = await self.metadata_storage.list()
        return [Document.model_validate_json(record) for record in records]

    async def get(self, document_id: str) -> Document:
        """Get one document record by its ID."""
        try:
            record = await self.metadata_storage.get(document_id)
        except KeyError:
            raise DocumentNotFoundError(document_id) from None
        return Document.model_validate_json(record)

    async def delete(self, document_id: str) -> bool:
        """Delete the file, its SQL tables, and its metadata when it exists."""
        await self.get(document_id)

        await self.file_storage.delete(document_id)
        if self.sql_storage:
            await self.sql_storage.delete_document(document_id)
        await self.metadata_storage.delete(document_id)

        return True
