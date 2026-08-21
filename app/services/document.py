from fastapi import UploadFile
from pydantic import ValidationError

from app.errors.document import DocumentNotFoundError
from app.models.document import Document
from app.store_file.base import FileStorage
from app.store_sql2.base import SqlStorage
from app.store_vector.base import VectorStorage
from app.utils.chunks import deserialize_chunk

DOCUMENT_METADATA_COLLECTION = "documents"


# class DocumentService:
#     """Coordinates file, metadata, and spreadsheet-table lifecycle operations.

#     Flow: upload() > FileStorage.save() > MetadataStorage.save()
#     """

#     def __init__(
#         self,
#         file_storage: FileStorage,
#         sql_storage: SqlStorage,
#         vector_storage: VectorStorage,
#     ):
#         self.file_storage = file_storage
#         self.sql_storage = sql_storage
#         self.vector_storage = vector_storage

#     async def upload(self, file: UploadFile) -> Document:
#         """Save an upload and register the resulting document metadata."""
#         # Save the file first because metadata needs its generated document details.
#         document = await self.file_storage.save(file)

#         # Serialization belongs to this caller; the metadata store only persists JSON.
#         await self.metadata_storage.upsert(
#             DOCUMENT_METADATA_COLLECTION, document.id, document.model_dump_json()
#         )

#         return document

#     async def get_all(self) -> list[Document]:
#         """List every saved document record."""
#         records = await self.metadata_storage.list()
#         documents = []
#         for record in records:
#             try:
#                 documents.append(Document.model_validate_json(record))
#             except ValidationError:
#                 # Generic metadata storage also contains chunk records.
#                 continue
#         return documents

#     async def get(self, document_id: str) -> Document:
#         """Get one document record by its ID."""
#         try:
#             record = await self.metadata_storage.get(document_id)
#         except KeyError:
#             raise DocumentNotFoundError(document_id) from None
#         return Document.model_validate_json(record)

#     async def read_bytes(self, document_id: str) -> bytes:
#         return await self.file_storage.read_bytes(document_id)

#     async def delete(self, document_id: str) -> bool:
#         """Delete the file, its SQL tables, and its metadata when it exists."""
#         await self.get(document_id)

#         chunk_ids = await self._chunk_ids_for_document(document_id)
#         if self.vector_storage:
#             await self.vector_storage.delete(chunk_ids)
#         for chunk_id in chunk_ids:
#             await self.metadata_storage.delete(chunk_id)
#         await self.file_storage.delete(document_id)
#         if self.sql_storage:
#             await self.sql_storage.delete_document(document_id)
#         await self.metadata_storage.delete(document_id)

#         return True

#     async def _chunk_ids_for_document(self, document_id: str) -> list[str]:
#         """Find chunk records owned by a document in generic metadata storage."""
#         chunk_ids = []
#         for record in await self.metadata_storage.list():
#             try:
#                 chunk = deserialize_chunk(record)
#             except ValueError:
#                 continue
#             if chunk.document.id == document_id:
#                 chunk_ids.append(chunk.id)
#         return chunk_ids
