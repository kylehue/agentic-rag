from collections.abc import Sequence
from io import BytesIO
from pathlib import Path
from uuid import uuid4

from app.core.config import settings
from app.embedders.base import Embedder
from app.llm.base import LLMProvider
from app.models.chunk import IngestedChunk, RetrievedChunk
from app.plugin.context import EmittedFile, IngestionContext
from app.plugin.hooks import HookBus, HOOK_CHUNK_SAVED, HOOK_FILE_EMITTED
from app.store_file.base import FileStorage
from app.store_sql.base import SqlStorage
from app.store_vector.base import VectorStorage


class IngestionRuntime:
    """Per-ingestion actions available to a plugin.

    Plugins persist their own chunks through this runtime. It also records
    files a plugin emits so the ingestion service can run them back through
    the pipeline (subprocess) with the plugins that accept them.
    """

    def __init__(
        self,
        *,
        context: IngestionContext,
        hooks: HookBus,
        llm: LLMProvider,
        embedder: Embedder,
        vector_storage: VectorStorage,
        sql_storage: SqlStorage,
        file_storage: FileStorage,
    ) -> None:
        self._context = context
        self._hooks = hooks
        self._llm = llm
        self._embedder = embedder
        self._vector_storage = vector_storage
        self._sql_storage = sql_storage
        self._file_storage = file_storage
        self._emitted: list[EmittedFile] = []

    @property
    def context(self) -> IngestionContext:
        return self._context

    @property
    def hooks(self) -> HookBus:
        return self._hooks

    @property
    def llm(self) -> LLMProvider:
        return self._llm

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed texts into vectors."""
        return await self._embedder.embed_documents(list(texts))

    async def add_vectors(
        self,
        chunk_ids: Sequence[str],
        embeddings: Sequence[Sequence[float]],
    ) -> None:
        """Store chunk vectors in the vector database."""
        await self._vector_storage.add(
            list(chunk_ids),
            [list(embedding) for embedding in embeddings],
        )

    async def save_chunk_file(self, chunk: IngestedChunk) -> str | None:
        """Store the chunk's file in file storage and record it in chunk metadata."""
        if not (
            chunk.file_bytes
            and chunk.file_filename
            and chunk.file_content_type
        ):
            return None

        file_path = await self._file_storage.upload(
            file=BytesIO(chunk.file_bytes),
            file_content_type=chunk.file_content_type,
            file_filename=f"{uuid4()}{Path(chunk.file_filename).suffix}",
            file_dir="chunk_files/",
        )

        chunk.metadata["chunk_file_path"] = file_path
        chunk.metadata["chunk_file_filename"] = chunk.file_filename
        chunk.metadata["chunk_file_content_type"] = chunk.file_content_type

        return file_path

    async def save_chunk_metadata(self, chunk: IngestedChunk) -> None:
        """Persist a chunk's `chunk_*` metadata to the chunk table."""
        await self._sql_storage.upsert(
            settings.CHUNK_TABLE_NAME,
            [
                {
                    "chunk_id": chunk.chunk_id,
                    "source_id": self._context.source_id,
                    "plugin": chunk.plugin,
                    "text": chunk.text,
                    "metadata": {
                        key: value
                        for key, value in chunk.metadata.items()
                        if key.startswith("chunk_")
                    },
                }
            ],
            ["id"],
        )

    async def save_chunks(self, chunks: Sequence[IngestedChunk]) -> None:
        """Persist chunks: embed, vectorize, store chunk file, store metadata."""
        chunks = list(chunks)
        if not chunks:
            return

        for chunk in chunks:
            chunk.metadata.setdefault("chunk_source_id", self._context.source_id)
            if self._context.parent_source_id is not None:
                chunk.metadata.setdefault(
                    "chunk_parent_source_id",
                    self._context.parent_source_id,
                )

        embeddings = await self.embed([chunk.text for chunk in chunks])
        await self.add_vectors(
            [chunk.chunk_id for chunk in chunks],
            embeddings,
        )

        for chunk in chunks:
            await self.save_chunk_file(chunk)
            await self.save_chunk_metadata(chunk)
            await self._hooks.emit(
                HOOK_CHUNK_SAVED,
                context=self._context,
                chunk=chunk,
            )

    async def save_chunk(self, chunk: IngestedChunk) -> None:
        """Persist a single chunk."""
        await self.save_chunks([chunk])

    async def save_source(
        self,
        file_bytes: bytes,
        filename: str,
        content_type: str,
    ) -> str:
        """Store the source file in file storage and record it in the documents table."""
        file_path = await self._file_storage.upload(
            file=BytesIO(file_bytes),
            file_content_type=content_type,
            file_filename=f"{uuid4()}{Path(filename).suffix}",
            file_dir="documents/",
        )

        await self._sql_storage.upsert(
            settings.DOCUMENT_METADATA_TABLE_NAME,
            [
                {
                    "source_id": self._context.source_id,
                    "file_path": file_path,
                    "file_content_type": content_type,
                    "file_filename": Path(file_path).name,
                    "file_orig_filename": filename,
                }
            ],
            ["id"],
        )

        return file_path

    async def emit_file(
        self,
        filename: str,
        content_type: str,
        file_bytes: bytes,
    ) -> EmittedFile:
        """Emit a file so it is ingested as a subprocess by its owning plugin."""
        emitted = EmittedFile(
            filename=filename,
            content_type=content_type,
            file_bytes=file_bytes,
            source_id=self._context.source_id,
        )
        self._emitted.append(emitted)
        await self._hooks.emit(
            HOOK_FILE_EMITTED,
            context=self._context,
            emitted_file=emitted,
        )
        return emitted

    def take_emitted_files(self) -> list[EmittedFile]:
        """Pop all emitted files. Called by the ingestion service."""
        emitted = self._emitted
        self._emitted = []
        return emitted


class FinalizeRuntime:
    """Per-retrieval actions available to a plugin's finalizer."""

    def __init__(
        self,
        *,
        llm: LLMProvider,
        hooks: HookBus,
    ) -> None:
        self._llm = llm
        self._hooks = hooks

    @property
    def llm(self) -> LLMProvider:
        return self._llm

    @property
    def hooks(self) -> HookBus:
        return self._hooks
