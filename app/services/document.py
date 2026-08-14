from fastapi import UploadFile

from app.models.document import Document
from app.store_file.base import FileStorage
from app.store_metadata.base import MetadataStorage
from app.store_sql.base import SqlStorage


class DocumentService:
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
        # save file
        document = await self.file_storage.save(file)

        # save metadata
        await self.metadata_storage.save(document)

        return document

    async def list(self) -> list[Document]:
        return await self.metadata_storage.list()

    async def get(self, document_id: str) -> Document | None:
        return await self.metadata_storage.get(document_id)

    async def delete(self, document_id: str) -> bool:
        document = await self.metadata_storage.get(document_id)

        if document is None:
            return False

        await self.file_storage.delete(document_id)
        if self.sql_storage:
            await self.sql_storage.delete_document(document_id)
        await self.metadata_storage.delete(document_id)

        return True
