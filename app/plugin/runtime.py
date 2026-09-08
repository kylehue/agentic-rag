from app.embedders.base import Embedder
from app.llm.base import LLMProvider
from app.plugin.context import IngestionContext, IngestionFile, RetrievalContext
from app.plugin.hooks import FileEmittedPayload, HookBus
from app.store_file.base import FileStorage
from app.store_sql.base import SqlStorage
from app.store_vector.base import VectorStorage


class IngestionRuntime:
    """Per-ingestion-run state and actions available to plugins.

    Created by the IngestionService for each ingestion (including
    subprocessed files). Plugins use it to reach the shared services
    (`llm`, `embedder`, the storages, `hooks`) and to emit files for
    subprocess. Chunk and source persistence stays with the ingestion
    service; the storages are exposed for plugins that need to read or
    write other content.
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
        self._emitted: list[IngestionFile] = []

    @property
    def context(self) -> IngestionContext:
        return self._context

    @property
    def hooks(self) -> HookBus:
        return self._hooks

    @property
    def llm(self) -> LLMProvider:
        return self._llm

    @property
    def embedder(self) -> Embedder:
        return self._embedder

    @property
    def vector_storage(self) -> VectorStorage:
        return self._vector_storage

    @property
    def sql_storage(self) -> SqlStorage:
        return self._sql_storage

    @property
    def file_storage(self) -> FileStorage:
        return self._file_storage

    async def emit_file(
        self,
        filename: str,
        content_type: str,
        file_bytes: bytes,
        description: str | None = None,
    ) -> IngestionFile:
        """Emit a file so it is ingested as a subprocess by the plugins that accept it.

        ``description`` is optional context about the emitted file; plugins
        handling it read it back as ``context.file.description``.
        """
        # The emitted file gets its own auto-generated source_id; its
        # lineage to this run is established by the service (parent_file).
        emitted = IngestionFile(
            filename=filename,
            content_type=content_type,
            file_bytes=file_bytes,
            description=description,
        )
        self._emitted.append(emitted)
        await self._hooks.trigger(
            "file_emitted",
            FileEmittedPayload(
                context=self._context,
                emitted_file=emitted,
                runtime=self,
            ),
        )
        return emitted

    def pop_emitted_files(self) -> list[IngestionFile]:
        """Pop all emitted files. Called by the ingestion service."""
        emitted = self._emitted
        self._emitted = []
        return emitted


class RetrievalRuntime:
    """Per-retrieval-run state available to plugins during retrieval.

    Created by the RetrievalService for each retrieve() call. Finalize
    handlers use it to reach the retrieval context (the user query), the
    shared LLM, and the hook bus.
    """

    def __init__(
        self,
        *,
        context: RetrievalContext,
        hooks: HookBus,
        llm: LLMProvider,
    ) -> None:
        self._context = context
        self._hooks = hooks
        self._llm = llm

    @property
    def context(self) -> RetrievalContext:
        return self._context

    @property
    def hooks(self) -> HookBus:
        return self._hooks

    @property
    def llm(self) -> LLMProvider:
        return self._llm
