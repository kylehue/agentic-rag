import asyncio
from dataclasses import replace
from io import BytesIO
from uuid import uuid4

from fastapi import UploadFile
from sqlalchemy import JSON, Column, Integer, String, Text
from unstructured.partition.auto import partition

from app.core.config import settings
from app.embedders.base import Embedder
from app.errors.document import InvalidDocumentError
from app.llm.base import LLMProvider
from app.models.chunk import IngestedChunk
from app.plugin.context import IngestionContext
from app.plugin.hooks import (
    HookBus,
    HOOK_FILE_SUBPROCESSED,
    HOOK_INGESTION_COMPLETED,
    HOOK_INGESTION_STARTED,
    HOOK_PLUGINS_SELECTED,
)
from app.plugin.registry import PluginRegistry
from app.plugin.runtime import IngestionRuntime
from app.store_file.base import FileStorage
from app.store_sql.base import SqlStorage
from app.store_vector.base import VectorStorage

MAX_SUBPROCESS_DEPTH = 5


class IngestionService:
    """Runs a file (and any files emitted during processing) through the plugin pipeline.

    The service owns orchestration only: it offers the file to every
    registered plugin, lets the plugins decide whether they want to manage
    it, builds the context and runtime, and re-ingests emitted files. All
    chunk persistence is handled by the plugins through the runtime.
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
            raise InvalidDocumentError(
                f"Document '{source_filename}' is not supported. Try adding a plugin."
            )

        if any(plugin.uses_elements for plugin in plugins):
            context = replace(
                context,
                elements=await asyncio.to_thread(
                    partition,
                    file=BytesIO(source_bytes),
                    file_filename=source_filename,
                    content_type=source_content_type,
                    strategy="hi_res",
                    infer_table_structure=True,  # Keep tables as structured HTML
                ),
            )

        runtime = IngestionRuntime(
            context=context,
            hooks=self._hooks,
            llm=self._llm,
            embedder=self._embedder,
            vector_storage=self._vector_storage,
            sql_storage=self._sql_storage,
            file_storage=self._file_storage,
        )

        await self._hooks.emit(HOOK_INGESTION_STARTED, context=context)
        await self._hooks.emit(
            HOOK_PLUGINS_SELECTED,
            context=context,
            plugins=plugins,
        )

        await runtime.save_source(
            source_bytes,
            source_filename,
            source_content_type,
        )

        all_chunks: list[IngestedChunk] = []

        for plugin in plugins:
            all_chunks.extend(await plugin.process(context, runtime))

        # recursively ingest the emitted files
        for emitted_file in runtime.take_emitted_files():
            emitted_chunks = await self.ingest_bytes(
                emitted_file.file_bytes,
                emitted_file.filename,
                emitted_file.content_type,
                parent_source_id=context.source_id,
                depth=depth + 1,
            )

            await self._hooks.emit(
                HOOK_FILE_SUBPROCESSED,
                emitted_file=emitted_file,
                chunks=emitted_chunks,
            )

            all_chunks.extend(emitted_chunks)

        await self._hooks.emit(
            HOOK_INGESTION_COMPLETED,
            context=context,
            chunks=all_chunks,
        )

        return all_chunks
