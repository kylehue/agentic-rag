import asyncio
import logging
from collections.abc import Sequence
from io import BytesIO
from pathlib import Path
from uuid import uuid4

from fastapi import UploadFile
from sqlalchemy import JSON, Column, Integer, String, Text

from app.core.config import settings
from app.embedders.base import Embedder
from app.errors.document import InvalidDocumentError
from app.llm.base import LLMProvider
from app.models.chunk import IngestedChunk
from app.plugin.context import IngestionContext
from app.plugin.hooks import (
    FileSubprocessedPayload,
    HookBus,
    IngestionCompletedPayload,
    IngestionProcessPayload,
    IngestionStartedPayload,
)
from app.plugin.registry import PluginRegistry
from app.plugin.runtime import IngestionRuntime
from app.store_file.base import FileStorage
from app.store_sql.base import SqlStorage
from app.store_vector.base import VectorStorage

MAX_SUBPROCESS_DEPTH = 5

logger = logging.getLogger(__name__)


class IngestionService:
    """Runs a file (and any files emitted during processing) through the hook pipeline.

    The service owns all persistence: source files, chunk files, chunk
    records, and vectors. Plugins generate chunks by handling the
    `ingestion.process` hook and may emit files for subprocess.
    """

    def __init__(
        self,
        *,
        hooks: HookBus,
        registry: PluginRegistry,
        llm: LLMProvider,
        embedder: Embedder,
        vector_storage: VectorStorage,
        sql_storage: SqlStorage,
        file_storage: FileStorage,
    ) -> None:
        self._hooks = hooks
        self._registry = registry
        self._llm = llm
        self._embedder = embedder
        self._vector_storage = vector_storage
        self._sql_storage = sql_storage
        self._file_storage = file_storage

    async def initialize(self) -> None:
        await self._sql_storage.ensure_table(
            settings.CHUNK_TABLE_NAME,
            [
                Column("id", Integer, primary_key=True, autoincrement=True),
                Column("chunk_id", String, nullable=False, unique=True),
                Column("source_id", String, nullable=False),
                Column("plugin", String, nullable=False),
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

    async def ingest(self, file_upload: UploadFile) -> list[IngestedChunk]:
        """Ingest one uploaded file."""
        if not file_upload.filename or not file_upload.content_type:
            raise InvalidDocumentError(
                "Invalid document. File name or content type is undefined."
            )

        source_bytes = await file_upload.read()

        return await self.ingest_bytes(
            source_bytes,
            file_upload.filename,
            file_upload.content_type,
        )

    async def ingest_bytes(
        self,
        source_bytes: bytes,
        source_filename: str,
        source_content_type: str,
        parent_source_id: str | None = None,
        parent_source_filename: str | None = None,
        depth: int = 0,
    ) -> list[IngestedChunk]:
        """Ingest raw bytes, including files emitted by plugins."""
        if depth >= MAX_SUBPROCESS_DEPTH:
            raise InvalidDocumentError("Document subprocess depth exceeded.")

        context = IngestionContext(
            source_id=str(uuid4()),
            source_filename=source_filename,
            source_content_type=source_content_type,
            source_bytes=source_bytes,
            parent_source_id=parent_source_id,
        )

        plugins = self._registry.accepting_plugins(context)

        if not plugins:
            if depth == 0:
                raise InvalidDocumentError(
                    f"No plugin accepted document '{source_filename}'."
                )
            else:
                logger.warning(
                    f"The document '{parent_source_filename}' emitted files that weren't accepted by any plugins."
                )
                return []  # avoid ingestion

        runtime = IngestionRuntime(
            context=context,
            hooks=self._hooks,
            llm=self._llm,
        )

        await self._hooks.emit(
            "ingestion_started",
            IngestionStartedPayload(context=context, runtime=runtime),
        )

        await self._save_source(
            context, source_bytes, source_content_type, source_filename
        )

        # Plugins generate chunks by handling the process hook and may
        # emit files along the way.
        results = await self._hooks.emit(
            "ingestion_process",
            IngestionProcessPayload(context=context, runtime=runtime),
        )

        chunks: list[IngestedChunk] = []
        for result in results:
            if result:
                chunks.extend(result)

        await self._save_chunks(context, chunks)

        all_chunks = list(chunks)

        for emitted_file in runtime.take_emitted_files():
            emitted_chunks = await self.ingest_bytes(
                emitted_file.file_bytes,
                emitted_file.filename,
                emitted_file.content_type,
                parent_source_id=context.source_id,
                parent_source_filename=parent_source_filename or source_filename,
                depth=depth + 1,
            )

            await self._hooks.emit(
                "file_subprocessed",
                FileSubprocessedPayload(
                    emitted_file=emitted_file,
                    chunks=emitted_chunks,
                    runtime=runtime,
                ),
            )

            all_chunks.extend(emitted_chunks)

        if not all_chunks:
            logger.warning(
                "Ingestion of '%s' produced no chunks; the source file was "
                "stored but nothing was indexed. Verify that a plugin with "
                "an 'ingestion_process' handler accepts this document.",
                source_filename,
            )

        await self._hooks.emit(
            "ingestion_completed",
            IngestionCompletedPayload(
                context=context,
                chunks=all_chunks,
                runtime=runtime,
            ),
        )

        return all_chunks

    async def _save_source(
        self,
        context: IngestionContext,
        source_bytes: bytes,
        source_content_type: str,
        source_filename: str,
    ) -> str:
        """Store the source file and record it in the documents table."""
        file_path = await self._file_storage.upload(
            file=BytesIO(source_bytes),
            file_content_type=source_content_type,
            file_filename=f"{uuid4()}{Path(source_filename).suffix}",
            file_dir="documents/",
        )

        await self._sql_storage.upsert(
            settings.DOCUMENT_METADATA_TABLE_NAME,
            [
                {
                    "source_id": context.source_id,
                    "file_path": file_path,
                    "file_content_type": source_content_type,
                    "file_filename": Path(file_path).name,
                    "file_orig_filename": source_filename,
                }
            ],
            ["id"],
        )

        return file_path

    async def _save_chunks(
        self,
        context: IngestionContext,
        chunks: Sequence[IngestedChunk],
    ) -> None:
        """Embed chunks, store chunk files, and persist chunk records.

        Chunk metadata is saved as-is, plus the storage paths and lineage
        stamped by the service.
        """
        if not chunks:
            return

        embeddings = await self._embedder.embed_documents(
            [chunk.text for chunk in chunks]
        )
        await self._vector_storage.add(
            [chunk.chunk_id for chunk in chunks],
            embeddings,
        )

        rows: list[dict] = []

        for chunk in chunks:
            # Stamp storage paths and lineage on the chunk so the same
            # metadata is returned by the API and persisted.
            if context.parent_source_id is not None:
                chunk.metadata["parent_source_id"] = context.parent_source_id

            if chunk.file_bytes and chunk.file_filename and chunk.file_content_type:
                file_path = await self._file_storage.upload(
                    file=BytesIO(chunk.file_bytes),
                    file_content_type=chunk.file_content_type,
                    file_filename=f"{uuid4()}{Path(chunk.file_filename).suffix}",
                    file_dir="chunk_files/",
                )
                chunk.metadata["file_path"] = file_path

            rows.append(
                {
                    "chunk_id": chunk.chunk_id,
                    "source_id": context.source_id,
                    "plugin": chunk.plugin,
                    "text": chunk.text,
                    "metadata": chunk.metadata,
                }
            )

        await self._sql_storage.upsert(
            settings.CHUNK_TABLE_NAME,
            rows,
            ["id"],
        )
