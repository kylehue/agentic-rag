import logging
from collections.abc import Callable, Sequence
from dataclasses import replace
from io import BytesIO
from pathlib import Path
from typing import Any
from uuid import uuid4

from sqlalchemy import ColumnElement, Table

from app.database import CHUNK_TABLE_NAME, DOCUMENT_METADATA_TABLE_NAME
from app.embedders.base import ImageEmbedder, TextEmbedder
from app.errors.document import InvalidDocumentError
from app.ingest.events import ProgressEmitter
from app.llm.base import LLMProvider
from app.models.chunk import (
    IngestedChunk,
    IngestedImageChunk,
    IngestedTextChunk,
)
from app.models.content import ImageContent
from app.plugin.context import IngestionContext, IngestionFile
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
        registry: PluginRegistry,
        llm: LLMProvider,
        text_embedder: TextEmbedder,
        image_embedder: ImageEmbedder,
        text_vector_storage: VectorStorage,
        image_vector_storage: VectorStorage,
        sql_storage: SqlStorage,
        file_storage: FileStorage,
    ) -> None:
        self._registry = registry
        self._llm = llm
        self._text_embedder = text_embedder
        self._image_embedder = image_embedder
        self._text_vector_storage = text_vector_storage
        self._image_vector_storage = image_vector_storage
        self._sql_storage = sql_storage
        self._file_storage = file_storage

    async def ingest(
        self,
        file: IngestionFile,
        chat_id: str | None = None,
        emitter: ProgressEmitter | None = None,
    ) -> list[IngestedChunk]:
        """Ingest a file (and everything its plugins emit) into `chat_id`.

        The whole emission tree is stamped with the chat id: it becomes the
        group the retrieval layer can fetch natively. When `emitter` is given
        (a queued ingest), pipeline stages and plugin states are reported to
        it as they happen; otherwise nothing is reported.
        """

        pending_sources: list[tuple[IngestionFile, bool]] = []
        pending_chunks: list[tuple[IngestionContext, IngestedChunk]] = []
        origin_file = file

        async def _ingest(
            file: IngestionFile,
            parent_file: IngestionFile | None,
            depth: int,
            inherited_metadata: dict[str, Any],
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

            ``inherited_metadata`` is the merged metadata of every chunk
            produced above this file in the emission tree. This file's chunks
            inherit it (their own keys win on a collision), and the merged
            result is passed down to the files they emit.
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
                chat_id=chat_id,
            )

            runtime = IngestionRuntime(
                context=context,
                registry=self._registry,
                llm=self._llm,
                text_embedder=self._text_embedder,
                text_vector_storage=self._text_vector_storage,
                sql_storage=self._sql_storage,
                file_storage=self._file_storage,
                emitter=emitter,
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

            await self._registry.ingestion_started(context, runtime)

            # Every file that goes through ingestion is stored exactly once.
            # The flag marks user uploads so they can be told apart from
            # emitted files in the documents table.
            pending_sources.append((file, parent_file is None))

            # Plugins generate chunks here and may emit files along the way.
            own_chunks = await self._registry.ingestion_process(context, runtime)

            # This file's chunks inherit the metadata of the chunks above
            # them in the emission tree; on a key collision, this file's own
            # value wins. The merged result is what their children inherit.
            merged_own: list[IngestedChunk] = []
            children_metadata = dict(inherited_metadata)
            for chunk in own_chunks:
                merged = {**inherited_metadata, **chunk.metadata}
                merged_own.append(replace(chunk, metadata=merged))
                children_metadata.update(merged)

            # Recurse into the emitted files in emission order, collecting each
            # subtree and reporting it back through the subprocess hook.
            child_chunks: list[IngestedChunk] = []
            for emitted in runtime.pop_emitted_files():
                _, _, subtree = await _ingest(
                    file=emitted,
                    parent_file=file,
                    depth=depth + 1,
                    inherited_metadata=children_metadata,
                )
                child_chunks.extend(subtree)
                await self._registry.file_subprocessed(
                    emitted, subtree, context, runtime
                )

            all_chunks = [*merged_own, *child_chunks]

            if not all_chunks:
                logger.warning(
                    "Ingestion of '%s' produced no chunks; nothing was indexed. "
                    "Verify that a plugin with an 'ingestion_process' handler "
                    "accepts this document and generates chunks for it.",
                    file.filename,
                )

            # This file (and its whole emitted subtree) is processed. Nothing
            # is committed yet; ingestion_completed fires once, at the end.
            await self._registry.file_completed(all_chunks, context, runtime)

            pending_chunks.extend((context, chunk) for chunk in merged_own)

            return context, runtime, all_chunks

        if emitter:
            emitter.stage(file.filename, "processing")

        top_context, top_runtime, chunks = await _ingest(
            file=file,
            parent_file=None,
            depth=0,
            inherited_metadata={},
        )

        if emitter:
            emitter.stage(file.filename, "saving")

        await self._commit(
            pending_sources,
            pending_chunks,
            chat_id,
            file.filename,
            emitter,
        )

        # The very end: every file processed and everything committed.
        await self._registry.ingestion_completed(chunks, top_context, top_runtime)

        return chunks

    async def delete_file(
        self, origin_source_id: str, chat_id: str | None = None
    ) -> int:
        """Reverse ingestion: remove a file's whole emission tree.

        Deletes the file's chunks (located by ``origin_source_id``), their
        vectors, the SQL chunk and document records, and the stored file
        bytes -- for the origin file and every file it emitted. ``chat_id``,
        when given, bounds the deletion to that chat, so one chat's files
        can never be removed through another. Returns the number of files
        (documents) removed; 0 means the file was not found in the chat.
        """
        chunk_condition = self._chat_scoped(
            lambda t: t.c.origin_source_id == origin_source_id, chat_id
        )
        chunk_rows = list(
            await self._sql_storage.get_all(
                CHUNK_TABLE_NAME, condition=chunk_condition
            )
        )
        chunk_ids = [row["chunk_id"] for row in chunk_rows]
        # Every file in the tree is either the origin or the source of a chunk.
        source_ids = {row["source_id"] for row in chunk_rows}
        source_ids.add(origin_source_id)

        if chunk_ids:
            # Image chunks live in the image collection; text chunks in the text
            # one. Delete from both (each ignores ids it does not hold).
            await self._text_vector_storage.delete(chunk_ids)
            await self._image_vector_storage.delete(chunk_ids)
        if chunk_rows:
            await self._sql_storage.delete(
                CHUNK_TABLE_NAME, condition=chunk_condition
            )

        doc_condition = self._chat_scoped(
            lambda t: t.c.source_id.in_(source_ids), chat_id
        )
        doc_rows = list(
            await self._sql_storage.get_all(
                DOCUMENT_METADATA_TABLE_NAME, condition=doc_condition
            )
        )
        for doc in doc_rows:
            await self._file_storage.delete(doc["file_path"])
        if doc_rows:
            await self._sql_storage.delete(
                DOCUMENT_METADATA_TABLE_NAME, condition=doc_condition
            )

        return len(doc_rows)

    async def delete_chat(self, chat_id: str) -> None:
        """Reverse ingestion for a whole chat: remove every chunk, vector,
        document record, and stored file that belongs to it."""
        chunk_condition = lambda t: t.c.chat_id == chat_id
        chunk_rows = list(
            await self._sql_storage.get_all(
                CHUNK_TABLE_NAME, condition=chunk_condition
            )
        )
        chunk_ids = [row["chunk_id"] for row in chunk_rows]
        if chunk_ids:
            # Delete from both collections (image + text); each ignores ids
            # it does not hold.
            await self._text_vector_storage.delete(chunk_ids)
            await self._image_vector_storage.delete(chunk_ids)
        if chunk_rows:
            await self._sql_storage.delete(
                CHUNK_TABLE_NAME, condition=chunk_condition
            )

        doc_condition = lambda t: t.c.chat_id == chat_id
        doc_rows = list(
            await self._sql_storage.get_all(
                DOCUMENT_METADATA_TABLE_NAME, condition=doc_condition
            )
        )
        for doc in doc_rows:
            await self._file_storage.delete(doc["file_path"])
        if doc_rows:
            await self._sql_storage.delete(
                DOCUMENT_METADATA_TABLE_NAME, condition=doc_condition
            )

    async def get_file_metadata(self, source_id: str) -> dict | None:
        """The document record for one stored file, or None if absent."""
        return await self._sql_storage.get(
            DOCUMENT_METADATA_TABLE_NAME,
            condition=lambda t: t.c.source_id == source_id,
        )

    async def get_file_link(self, source_id: str) -> str | None:
        """The URL at which one stored file can be retrieved, or None if the
        file is absent."""
        document = await self.get_file_metadata(source_id)
        if document is None:
            return None
        return self._file_storage.create_link(document["file_path"])

    async def get_image(
        self, source_id: str, chat_id: str | None = None
    ) -> ImageContent | None:
        """The stored image for a source id, scoped to the chat.

        Loads the file bytes from file storage. Returns None if the file is
        absent or belongs to a different chat.
        """
        document = await self.get_file_metadata(source_id)
        if document is None:
            return None
        if chat_id is not None and document.get("chat_id") != chat_id:
            return None
        data = await self._file_storage.read_bytes(document["file_path"])
        return ImageContent(data=data, mime_type=document["file_content_type"])

    async def list_files(self, chat_id: str | None = None) -> list[dict]:
        """The origin files in the chat (what the user uploaded), excluding
        the files plugins emitted during ingestion (``is_origin``)."""
        condition = self._chat_scoped(lambda t: t.c.is_origin.is_(True), chat_id)
        return list(
            await self._sql_storage.get_all(
                DOCUMENT_METADATA_TABLE_NAME, condition=condition
            )
        )

    async def list_file_chunks(
        self, origin_source_id: str, chat_id: str | None = None
    ) -> list[dict]:
        """All chunks of a file's emission tree: the origin file's chunks plus
        every emitted descendant's, located via ``origin_source_id``."""
        condition = self._chat_scoped(
            lambda t: t.c.origin_source_id == origin_source_id, chat_id
        )
        return list(
            await self._sql_storage.get_all(
                CHUNK_TABLE_NAME, condition=condition
            )
        )

    @staticmethod
    def _chat_scoped(
        base: Callable[[Table], ColumnElement[bool]],
        chat_id: str | None,
    ) -> Callable[[Table], ColumnElement[bool]]:
        """AND a chat scope into a condition builder when ``chat_id`` is set."""
        if chat_id is None:
            return base

        def condition(table: Table) -> ColumnElement[bool]:
            return base(table) & (table.c.chat_id == chat_id)

        return condition

    async def _save_source(
        self,
        file: IngestionFile,
        is_origin: bool,
        chat_id: str | None,
    ) -> str:
        """Store a file and record it in the documents table.

        ``is_origin`` marks the file as a user upload (``True``) rather than
        a file emitted by a plugin (``False``).
        """
        file_path = await self._file_storage.upload(
            file=BytesIO(file.file_bytes),
            file_content_type=file.content_type,
            file_filename=f"{uuid4()}{Path(file.filename).suffix}",
            file_dir="documents/",
        )

        await self._sql_storage.upsert(
            DOCUMENT_METADATA_TABLE_NAME,
            [
                {
                    "source_id": file.source_id,
                    "file_path": file_path,
                    "file_content_type": file.content_type,
                    "file_filename": Path(file_path).name,
                    "file_orig_filename": file.filename,
                    "is_origin": is_origin,
                    "chat_id": chat_id,
                }
            ],
            ["id"],
        )

        return file_path

    async def _commit(
        self,
        pending_sources: Sequence[tuple[IngestionFile, bool]],
        pending_chunks: Sequence[tuple[IngestionContext, IngestedChunk]],
        chat_id: str | None,
        origin_filename: str,
        emitter: ProgressEmitter | None = None,
    ) -> None:
        """Persist everything collected during the walk, stamped with the
        chat the ingestion belongs to."""
        for file, is_origin in pending_sources:
            await self._save_source(file, is_origin, chat_id)

        if pending_chunks:
            if emitter:
                emitter.stage(
                    origin_filename, "embedding", chunk_count=len(pending_chunks)
                )
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
        # The chat id travels as vector metadata so the vector legs of
        # retrieval can be bounded to a chat at the index.
        def metas(records_subset):
            return [
                {"chat_id": context.chat_id} if context.chat_id else {}
                for context, _ in records_subset
            ]

        # Route chunks by modality: text chunks go to the text embedder and
        # the text collection; image chunks to the image embedder and the
        # image collection. The SQL records are the same either way.
        text_records = [
            (context, chunk)
            for context, chunk in records
            if isinstance(chunk, IngestedTextChunk)
        ]
        image_records = [
            (context, chunk)
            for context, chunk in records
            if isinstance(chunk, IngestedImageChunk)
        ]

        if text_records:
            text_chunks = [chunk for _, chunk in text_records]
            text_embeddings = await self._text_embedder.embed_text(
                [chunk.text for chunk in text_chunks]
            )
            await self._text_vector_storage.add(
                [chunk.chunk_id for chunk in text_chunks],
                text_embeddings,
                metadatas=metas(text_records),
            )

        if image_records:
            image_chunks = [chunk for _, chunk in image_records]
            # IngestedImageChunk guarantees a non-None image (checked in
            # __post_init__); the filter is only to satisfy the type checker.
            images = [
                chunk.image
                for chunk in image_chunks
                if chunk.image is not None
            ]
            image_embeddings = await self._image_embedder.embed_images(images)
            await self._image_vector_storage.add(
                [chunk.chunk_id for chunk in image_chunks],
                image_embeddings,
                metadatas=metas(image_records),
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
                # Image chunks carry no searchable text; their description is
                # a separate text chunk.
                "text": chunk.text if isinstance(chunk, IngestedTextChunk) else "",
                "metadata": chunk.metadata,
                "chat_id": context.chat_id,
                "key": chunk.key,
            }
            for context, chunk in records
        ]

        await self._sql_storage.upsert(
            CHUNK_TABLE_NAME,
            rows,
            ["id"],
        )
