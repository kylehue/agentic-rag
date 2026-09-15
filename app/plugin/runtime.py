from __future__ import annotations

from contextvars import ContextVar
from typing import TYPE_CHECKING, Any

from app.embedders.base import Embedder
from app.llm.base import LLMProvider
from app.plugin.context import IngestionContext, IngestionFile, RetrievalContext
from app.store_file.base import FileStorage
from app.store_sql.base import SqlStorage
from app.store_vector.base import VectorStorage

if TYPE_CHECKING:
    from app.ingest.events import ProgressEmitter
    from app.plugin.registry import PluginRegistry

# The plugin currently executing, set by the registry per fan-out task. Lets
# a plugin's `runtime.report_state(...)` be attributed without the plugin
# having to pass its own name.
current_plugin: ContextVar[str | None] = ContextVar("current_plugin", default=None)


class IngestionRuntime:
    """Per-ingestion-run state and actions available to plugins.

    Created by the IngestionService for each ingestion (including
    subprocessed files). Plugins use it to reach the shared services
    (`llm`, `embedder`, the storages) and to emit files for subprocess.
    Chunk and source persistence stays with the ingestion service; the
    storages are exposed for plugins that need to read or write other
    content.
    """

    def __init__(
        self,
        *,
        context: IngestionContext,
        registry: PluginRegistry,
        llm: LLMProvider,
        embedder: Embedder,
        vector_storage: VectorStorage,
        sql_storage: SqlStorage,
        file_storage: FileStorage,
        emitter: ProgressEmitter | None = None,
    ) -> None:
        self._context = context
        self._registry = registry
        self._llm = llm
        self._embedder = embedder
        self._vector_storage = vector_storage
        self._sql_storage = sql_storage
        self._file_storage = file_storage
        self._emitter = emitter
        self._emitted: list[IngestionFile] = []

    @property
    def context(self) -> IngestionContext:
        return self._context

    @property
    def registry(self) -> PluginRegistry:
        return self._registry

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
        await self._registry.file_emitted(emitted, self._context, self)
        return emitted

    def pop_emitted_files(self) -> list[IngestionFile]:
        """Pop all emitted files. Called by the ingestion service."""
        emitted = self._emitted
        self._emitted = []
        return emitted

    async def report_state(self, state: str, **detail: Any) -> None:
        """Report a fine-grained progress state for this run (e.g. a plugin
        announcing "partitioning"). No-op when the run has no progress
        emitter (direct/test ingestion). The plugin is attributed
        automatically by the registry."""
        if self._emitter is None:
            return
        self._emitter.plugin_state(
            file=self._context.file.filename,
            plugin=current_plugin.get(),
            state=state,
            **detail,
        )


class RetrievalRuntime:
    """Per-retrieval-run state available to plugins during retrieval.

    Created by the RetrievalService for each retrieve() call. Finalize
    handlers use it to reach the retrieval context (the user query) and
    the shared LLM.
    """

    def __init__(
        self,
        *,
        context: RetrievalContext,
        llm: LLMProvider,
    ) -> None:
        self._context = context
        self._llm = llm

    @property
    def context(self) -> RetrievalContext:
        return self._context

    @property
    def llm(self) -> LLMProvider:
        return self._llm
