from app.llm.base import LLMProvider
from app.plugin.context import EmittedFile, IngestionContext, RetrievalContext
from app.plugin.hooks import FileEmittedPayload, HookBus


class IngestionRuntime:
    """Per-ingestion-run state and actions available to plugins.

    Created by the IngestionService for each ingestion (including
    subprocessed files). Plugins use it to reach the shared services
    (`llm`, `hooks`) and to emit files for subprocess. It has no storage
    capabilities: the ingestion service persists everything.
    """

    def __init__(
        self,
        *,
        context: IngestionContext,
        hooks: HookBus,
        llm: LLMProvider,
    ) -> None:
        self._context = context
        self._hooks = hooks
        self._llm = llm
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

    async def emit_file(
        self,
        filename: str,
        content_type: str,
        file_bytes: bytes,
    ) -> EmittedFile:
        """Emit a file so it is ingested as a subprocess by the plugins that accept it."""
        emitted = EmittedFile(
            filename=filename,
            content_type=content_type,
            file_bytes=file_bytes,
            source_id=self._context.source_id,
        )
        self._emitted.append(emitted)
        await self._hooks.emit(
            "file_emitted",
            FileEmittedPayload(
                context=self._context,
                emitted_file=emitted,
                runtime=self,
            ),
        )
        return emitted

    def take_emitted_files(self) -> list[EmittedFile]:
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
