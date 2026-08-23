import asyncio
from collections.abc import Sequence
from io import BytesIO
from typing import Any
from uuid import uuid4

from fastapi import UploadFile
from sqlalchemy import JSON, Column, Integer, String, Text
from app.core.config import settings
from app.errors.document import InvalidDocumentError
from app.models.ingestion import ProcessorPayload
from app.processors.base import Processor
from app.embedders.base import Embedder
from app.store_file.base import FileStorage
from app.store_vector.base import VectorStorage
from app.store_sql.base import SqlStorage
from app.models.chunk import IngestedChunk, ChunkCategory
from app.llm.base import LLMProvider
from unstructured.partition.auto import partition
from pathlib import Path

from app.utils.conversion import filename_to_chunk_category


class IngestionService:
    """Turns an uploaded file into searchable chunks and optional SQL tables."""

    def __init__(
        self,
        llm: LLMProvider,
        embedder: Embedder,
        processor: Processor,
        file_storage: FileStorage,
        vector_storage: VectorStorage,
        sql_storage: SqlStorage,
    ):
        self._llm = llm
        self._embedder = embedder
        self._processor = processor
        self._file_storage = file_storage
        self._vector_storage = vector_storage
        self._sql_storage = sql_storage

    async def initialize(self) -> None:
        await self._sql_storage.ensure_table(
            settings.CHUNK_TABLE_NAME,
            [
                Column("id", Integer, primary_key=True, autoincrement=True),
                Column("chunk_id", String, nullable=False, unique=True),
                Column("source_id", String, nullable=False),
                Column("category", String, nullable=False),
                Column("text", Text, nullable=False),
                Column("metadata", JSON),
            ],
        )

        await self._sql_storage.ensure_table(
            settings.DOCUMENT_METADATA_TABLE_NAME,
            [
                Column("id", Integer, primary_key=True, autoincrement=True),
                Column("source_id", String, nullable=False, unique=True),
                Column("file_path", String, nullable=False),
                Column("file_content_type", String, nullable=False),
                Column("file_filename", String, nullable=False),
                Column("file_orig_filename", String, nullable=False),
            ],
        )

    async def ingest(self, file_upload: UploadFile):
        """Partition, process, embed, and save one uploaded document."""
        if not file_upload.filename or not file_upload.content_type:
            raise InvalidDocumentError(
                "Invalid document. File name or content type is undefined."
            )

        source_id = str(uuid4())  # used for chunk referencing
        source_bytes = await file_upload.read()
        source_content_type = file_upload.content_type
        source_filename = file_upload.filename

        # 1. Partition
        elements = await asyncio.to_thread(
            partition,
            file=BytesIO(source_bytes),
            file_filename=source_filename,
            content_type=source_content_type,
            strategy="hi_res",
            infer_table_structure=False,  # Keep tables as structured HTML, not jumbled text (false for now)
            extract_image_block_types=["Image"],  # Grab images found in the PDF
            extract_image_block_to_payload=True,  # Store images as base64 data you can actually use
        )

        # 2. Chunk
        processor_payload = ProcessorPayload(
            source_id=source_id,
            source_filename=source_filename,
            source_bytes=source_bytes,
            source_content_type=source_content_type,
            elements=elements,
            category=filename_to_chunk_category(source_filename),
        )
        chunks = await self._processor.process(processor_payload)

        # 3. Save to databases
        await self._save_chunk_to_vector_db(chunks)
        for chunk in chunks:
            await self._save_spreadsheet_chunk_to_sql_db(chunk)
            await self._save_chunk_to_file_db(chunk)

        await self._save_chunks_to_sql_db(source_id, chunks)
        await self._save_source_to_file_db(
            source_id=source_id,
            source_bytes=source_bytes,
            source_content_type=source_content_type,
            source_filename=source_filename,
        )

        return chunks

    async def _save_chunk_to_vector_db(self, chunks: Sequence[IngestedChunk]):
        """Embeds chunks and saves them to the vector database."""
        embeddings = await self._embedder.embed_documents(
            [chunk.text for chunk in chunks]
        )
        chunk_ids = [c.chunk_id for c in chunks]
        await self._vector_storage.add(chunk_ids, embeddings)

    async def _save_spreadsheet_chunk_to_sql_db(self, chunk: IngestedChunk) -> None:
        """Saves a spreadsheet chunk's table data to SQL database."""

        if chunk.category is not ChunkCategory.SPREADSHEET:
            return

        metadata = chunk.metadata

        table_name = metadata.get("sql_table_name")
        schema = metadata.get("sql_schema")
        rows = metadata.get("sql_rows")

        if not table_name or not schema:
            return

        # Create the table if it doesn't already exist
        await self._sql_storage.ensure_table(
            table_name,
            self._sql_storage.create_sql_columns_from_schema(schema),
        )

        # Nothing to insert
        if not rows:
            return

        # Insert/update the table data
        await self._sql_storage.upsert(
            table_name,
            rows,
            conflict_columns=[],
        )

    async def _save_chunk_to_file_db(self, chunk: IngestedChunk):
        """Saves a chunk's `file_bytes` to the file database when provided."""

        # Only save the chunk as file if bytes exist
        if (
            not chunk.file_bytes
            or not chunk.file_filename
            or not chunk.file_content_type
        ):
            return

        file_extension = Path(chunk.file_filename).suffix

        file_path = await self._file_storage.upload(
            file=BytesIO(chunk.file_bytes),
            file_content_type=chunk.file_content_type,
            file_filename=f"{uuid4()}{file_extension}",
            file_dir="chunk_files/",
        )

        chunk.metadata["chunk_file_path"] = file_path

    async def _save_chunks_to_sql_db(
        self,
        source_id: str,
        chunks: Sequence[IngestedChunk],
    ):
        """Saves chunks to the SQL database."""

        rows: list[dict[str, Any]] = []

        for chunk in chunks:
            cleaned_metadata = {
                key: value
                for key, value in chunk.metadata.items()
                if key.startswith("chunk_")
            }

            rows.append(
                {
                    "chunk_id": chunk.chunk_id,
                    "source_id": source_id,
                    "category": chunk.category.value,
                    "text": chunk.text,
                    "metadata": cleaned_metadata,
                }
            )

        await self._sql_storage.upsert(
            settings.CHUNK_TABLE_NAME,
            rows,
            ["id"],
        )

    async def _save_source_to_file_db(
        self,
        source_id: str,
        source_bytes: bytes,
        source_content_type: str,
        source_filename: str,
    ):
        source_extension = Path(source_filename).suffix
        source_new_filename = f"{uuid4()}{source_extension}"
        file_path = await self._file_storage.upload(
            file=BytesIO(source_bytes),
            file_content_type=source_content_type,
            file_filename=source_new_filename,
            file_dir="documents/",
        )
        await self._sql_storage.upsert(
            settings.DOCUMENT_METADATA_TABLE_NAME,
            [
                {
                    "source_id": source_id,
                    "file_path": file_path,
                    "file_content_type": source_content_type,
                    "file_filename": source_new_filename,
                    "file_orig_filename": source_filename,
                }
            ],
            ["id"],
        )
