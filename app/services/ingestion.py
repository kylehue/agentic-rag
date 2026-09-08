import logging
from collections.abc import Sequence
from io import BytesIO
from pathlib import Path
from uuid import uuid4

from sqlalchemy import JSON, Boolean, Column, Integer, String, Text

from app.core.config import settings
from app.embedders.base import Embedder
from app.errors.document import InvalidDocumentError
from app.llm.base import LLMProvider
from app.models.chunk import IngestedChunk
from app.plugin.context import IngestionContext, IngestionFile
from app.plugin.hooks import (
    FileCompletedPayload,
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
    """Runs a file (and any files emitted during processing) through the hook pipeline."""

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
                Column("parent_source_id", String, nullable=True),
                Column("origin_source_id", String, nullable=False),
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
                Column("is_origin", Boolean, nullable=False),
            ],
        )

    async def ingest(self, file: IngestionFile) -> list[IngestedChunk]:
        """Ingest a file and everything its plugins emit."""

        pending_sources: list[tuple[IngestionFile, bool]] = []
        pending_chunks: list[tuple[IngestionContext, IngestedChunk]] = []
        origin_file = file

        async def _ingest(
            file: IngestionFile,
            parent_file: IngestionFile | None,
            depth: int,
        ) -> tuple[IngestionContext, IngestionRuntime, list[IngestedChunk]]:
            """Process one file, then recurse into every file it emits.

            Returns the file's context, its runtime, and the chunks of the
            file's whole subtree (its own chunks plus every descendant's), so
            the parent can report the subtree back through the
            ``file_subprocessed`` hook. ``origin_file`` is the top-most file
            of the emission chain and is inherited unchanged by every emitted
            child. An empty chunk list marks a skipped file: no plugin
            accepted it, so it was neither stored nor processed (its runtime
            is discarded without firing any hooks).
            """
            if depth >= MAX_SUBPROCESS_DEPTH:
                raise InvalidDocumentError("Document subprocess depth exceeded.")

            if not file.filename or not file.content_type:
                raise InvalidDocumentError(
                    "Invalid document. File name or content type is undefined."
                )

            context = IngestionContext(
                file=file,
                origin_file=origin_file,
                parent_file=parent_file,
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

            plugins = self._registry.accepting_plugins(context)

            if not plugins:
                # only raise error if it's the origin file
                if parent_file is None:
                    raise InvalidDocumentError(
                        f"No plugin accepted document '{file.filename}'."
                    )
                logger.warning(
                    f"The document '{parent_file.filename}' emitted files that weren't accepted by any plugins."
                )
                return context, runtime, []  # avoid ingestion

            await self._hooks.trigger(
                "ingestion_started",
                IngestionStartedPayload(context=context, runtime=runtime),
            )

            # Every file that goes through ingestion is stored exactly once.
            # The flag marks user uploads so they can be told apart from
            # emitted files in the documents table.
            pending_sources.append((file, parent_file is None))

            # Plugins generate chunks here and may emit files along the way.
            results = await self._hooks.trigger(
                "ingestion_process",
                IngestionProcessPayload(context=context, runtime=runtime),
            )
            own_chunks = [chunk for result in results if result for chunk in result]

            # Recurse into the emitted files in emission order, collecting each
            # subtree and reporting it back through the subprocess hook.
            child_chunks: list[IngestedChunk] = []
            for emitted in runtime.pop_emitted_files():
                _, _, subtree = await _ingest(
                    file=emitted,
                    parent_file=file,
                    depth=depth + 1,
                )
                child_chunks.extend(subtree)
                await self._hooks.trigger(
                    "file_subprocessed",
                    FileSubprocessedPayload(
                        emitted_file=emitted,
                        chunks=subtree,
                        runtime=runtime,
                    ),
                )

            all_chunks = [*own_chunks, *child_chunks]

            if not all_chunks:
                logger.warning(
                    "Ingestion of '%s' produced no chunks; nothing was indexed. "
                    "Verify that a plugin with an 'ingestion_process' handler "
                    "accepts this document and generates chunks for it.",
                    file.filename,
                )

            # This file (and its whole emitted subtree) is processed. Nothing
            # is committed yet; ingestion_completed fires once, at the end.
            await self._hooks.trigger(
                "file_completed",
                FileCompletedPayload(
                    context=context,
                    chunks=all_chunks,
                    runtime=runtime,
                ),
            )

            pending_chunks.extend((context, chunk) for chunk in own_chunks)

            return context, runtime, all_chunks

        top_context, top_runtime, chunks = await _ingest(
            file=file,
            parent_file=None,
            depth=0,
        )

        await self._commit(pending_sources, pending_chunks)

        # The very end: every file processed and everything committed.
        await self._hooks.trigger(
            "ingestion_completed",
            IngestionCompletedPayload(
                context=top_context,
                chunks=chunks,
                runtime=top_runtime,
            ),
        )

        return chunks

    async def _save_source(
        self,
        file: IngestionFile,
        is_origin: bool,
    ) -> str:
        """Store a file and record it in the documents table.

        ``is_origin`` marks the file as a user upload (``True``) rather than a
        file emitted by a plugin (``False``).
        """
        file_path = await self._file_storage.upload(
            file=BytesIO(file.file_bytes),
            file_content_type=file.content_type,
            file_filename=f"{uuid4()}{Path(file.filename).suffix}",
            file_dir="documents/",
        )

        await self._sql_storage.upsert(
            settings.DOCUMENT_METADATA_TABLE_NAME,
            [
                {
                    "source_id": file.source_id,
                    "file_path": file_path,
                    "file_content_type": file.content_type,
                    "file_filename": Path(file_path).name,
                    "file_orig_filename": file.filename,
                    "is_origin": is_origin,
                }
            ],
            ["id"],
        )

        return file_path

    async def _commit(
        self,
        pending_sources: Sequence[tuple[IngestionFile, bool]],
        pending_chunks: Sequence[tuple[IngestionContext, IngestedChunk]],
    ) -> None:
        """Persist everything collected during the walk."""
        for file, is_origin in pending_sources:
            await self._save_source(file, is_origin)

        if pending_chunks:
            await self._save_chunk_records(pending_chunks)

    async def _save_chunk_records(
        self,
        records: Sequence[tuple[IngestionContext, IngestedChunk]],
    ) -> None:
        """Embed chunks and persist their records.

        ``records`` pairs each chunk with the context it was produced in, so
        the lineage columns (source / parent / origin) are stamped correctly
        even though chunks from several files are saved in one batch. Chunk
        metadata is saved as-is.
        """
        chunks = [chunk for _, chunk in records]

        embeddings = await self._embedder.embed_documents(
            [chunk.text for chunk in chunks]
        )
        await self._vector_storage.add(
            [chunk.chunk_id for chunk in chunks],
            embeddings,
        )

        rows = [
            {
                "chunk_id": chunk.chunk_id,
                "source_id": context.file.source_id,
                "parent_source_id": (
                    context.parent_file.source_id if context.parent_file else None
                ),
                "origin_source_id": context.origin_file.source_id,
                "plugin": chunk.plugin,
                "text": chunk.text,
                "metadata": chunk.metadata,
            }
            for context, chunk in records
        ]

        await self._sql_storage.upsert(
            settings.CHUNK_TABLE_NAME,
            rows,
            ["id"],
        )
