import asyncio
from io import BytesIO
from uuid import uuid4

from fastapi import UploadFile
from app.errors.document import InvalidDocumentError
from app.models.ingestion import ProcessorPayload
from app.processors.pipeline import process
from app.embedders.base import Embedder
from app.store_file.base import FileStorage
from app.store_vector.base import VectorStorage
from app.store_sql2.base import SqlStorage
from app.models.document import Document, DocumentCategory
from app.llm.base import LLMProvider
from unstructured.partition.auto import partition
from pathlib import Path
import pandas as pd

from app.utils.chunks import columns_from_document_chunk
from app.utils.file_type import detect_document_category

CHUNK_METADATA_COLLECTION = "chunks"
SPREADSHEET_CHUNK_DB_NAME = "spreadsheets"


class IngestionService:
    """Turns an uploaded file into searchable chunks and optional SQL tables.

    Flow: ingest() > process() > embed_documents() > add_documents()
    """

    def __init__(
        self,
        embedder: Embedder,
        llm: LLMProvider,
        file_storage: FileStorage,
        vector_storage: VectorStorage,
        sql_storage: SqlStorage,
    ):
        self.file_storage = file_storage
        self.embedder = embedder
        self.vector_storage = vector_storage
        self.llm = llm
        self.sql_storage = sql_storage

    async def ingest(self, file: UploadFile):
        """Partition, process, embed, and save one uploaded document."""
        if not file.filename or not file.content_type:
            raise InvalidDocumentError(
                "Invalid document. File name or content type is undefined."
            )

        file_id = str(uuid4())
        file_bytes = await file.read()
        file_content_type = file.content_type
        file_filename = file.filename

        # 1. Partition
        elements = await asyncio.to_thread(
            partition,
            file=BytesIO(file_bytes),
            file_filename=file_filename,
            content_type=file_content_type,
            strategy="hi_res",
            infer_table_structure=False,  # Keep tables as structured HTML, not jumbled text (false for now)
            extract_image_block_types=["Image"],  # Grab images found in the PDF
            extract_image_block_to_payload=True,  # Store images as base64 data you can actually use
        )

        # 2. Chunk
        processor_payload = ProcessorPayload(
            file_id=file_id,
            file_filename=file_filename,
            file_bytes=file_bytes,
            file_content_type=file.content_type,
            elements=elements,
            document_category=detect_document_category(file_filename),
            llm=self.llm,
        )
        chunks = await process(processor_payload)

        # 3. Save to databases
        # Vector DB
        embeddings = await self.embedder.embed_documents(
            [chunk.text for chunk in chunks]
        )
        chunk_ids = [c.id for c in chunks]
        await self.vector_storage.add(chunk_ids, embeddings)

        for chunk in chunks:
            await self._save_spreadsheet_chunk(chunk)
            await self._save_image_chunk(chunk)
            await self._save_chunk(chunk)

    async def _save_image_chunk(self, chunk: Document):
        """Saves image chunk to File DB and attaches file path to chunk metadata."""

        if chunk.category is not DocumentCategory.IMAGE:
            return

        if not chunk.file_bytes:
            return

        image_path = await self.file_storage.upload(
            file=BytesIO(chunk.file_bytes),
            file_filename=chunk.file_filename,
            file_content_type=chunk.file_content_type,
        )
        chunk.metadata["image_path"] = image_path

    async def _save_spreadsheet_chunk(self, chunk: Document) -> None:
        """Save spreadsheet chunk table data to SQL and attach its SQL location."""

        if chunk.category is not DocumentCategory.SPREADSHEET:
            return

        metadata = chunk.metadata

        table_name = metadata.get("sheet_name")

        if not isinstance(table_name, str) or not table_name:
            raise ValueError("Spreadsheet chunk is missing 'sheet_name'.")

        rows = metadata.get("rows")

        if not isinstance(rows, list):
            raise ValueError("Spreadsheet chunk is missing 'rows'.")

        # Create the table if it doesn't already exist
        await self.sql_storage.ensure_table(
            SPREADSHEET_CHUNK_DB_NAME,
            table_name,
            columns_from_document_chunk(chunk),
        )

        # Nothing to insert
        if not rows:
            chunk.metadata["sql_table"] = table_name
            chunk.metadata["sql_database"] = SPREADSHEET_CHUNK_DB_NAME
            return

        # Insert/update the table data
        await self.sql_storage.upsert(
            SPREADSHEET_CHUNK_DB_NAME,
            table_name,
            rows,
            conflict_columns=[],
        )

        # Attach the SQL DB location to chunk metadata
        chunk.metadata["sql_table"] = table_name
        chunk.metadata["sql_database"] = SPREADSHEET_CHUNK_DB_NAME

    async def _save_chunk(self, chunk: Document):
        """Saves chunk's data to SQL DB."""
        await self.sql_storage.upsert("chunks", "chunks", [], ["id"])
