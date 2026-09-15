from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from app.models.chunk import IngestedChunk, RetrievedChunk
from app.plugin.base import Plugin
from app.plugin.context import IngestionContext, IngestionFile, RetrievalContext
from app.plugin.runtime import current_plugin

if TYPE_CHECKING:
    from app.plugin.runtime import IngestionRuntime, RetrievalRuntime


class PluginRegistry:
    """Holds registered plugins and fans pipeline events out to them.

    Every event reaches every registered plugin, concurrently, in
    registration order; each plugin decides for itself whether to act. The
    two response events return per-plugin results the caller combines:
    ``ingestion_process`` returns the flattened chunks, and
    ``retrieval_finalize`` returns one optional replacement per plugin.
    """

    def __init__(self) -> None:
        self._plugins: list[Plugin] = []
        self._plugins_by_name: dict[str, Plugin] = {}

    def register(self, plugin: Plugin) -> None:
        if plugin.name in self._plugins_by_name:
            raise ValueError(f"Plugin name '{plugin.name}' is already registered.")

        self._plugins.append(plugin)
        self._plugins_by_name[plugin.name] = plugin

    def plugin_for(self, name: str) -> Plugin | None:
        return self._plugins_by_name.get(name)

    def accepting_plugins(self, context: IngestionContext) -> list[Plugin]:
        """All registered plugins that want to process this document."""
        return [plugin for plugin in self._plugins if plugin.accepts(context)]

    # --- ingestion events ---

    async def ingestion_started(
        self,
        context: IngestionContext,
        runtime: IngestionRuntime,
    ) -> None:
        await self._fan(lambda p: p.on_ingestion_started(context, runtime))

    async def ingestion_process(
        self,
        context: IngestionContext,
        runtime: IngestionRuntime,
    ) -> list[IngestedChunk]:
        results = await self._fan(lambda p: p.on_ingestion_process(context, runtime))
        return [chunk for result in results if result for chunk in result]

    async def file_emitted(
        self,
        emitted_file: IngestionFile,
        context: IngestionContext,
        runtime: IngestionRuntime,
    ) -> None:
        await self._fan(lambda p: p.on_file_emitted(emitted_file, context, runtime))

    async def file_subprocessed(
        self,
        emitted_file: IngestionFile,
        chunks: Sequence[IngestedChunk],
        context: IngestionContext,
        runtime: IngestionRuntime,
    ) -> None:
        await self._fan(
            lambda p: p.on_file_subprocessed(emitted_file, chunks, context, runtime)
        )

    async def file_completed(
        self,
        chunks: Sequence[IngestedChunk],
        context: IngestionContext,
        runtime: IngestionRuntime,
    ) -> None:
        await self._fan(lambda p: p.on_file_completed(chunks, context, runtime))

    async def ingestion_completed(
        self,
        chunks: Sequence[IngestedChunk],
        context: IngestionContext,
        runtime: IngestionRuntime,
    ) -> None:
        await self._fan(lambda p: p.on_ingestion_completed(chunks, context, runtime))

    # --- retrieval events ---

    async def retrieval_finalize(
        self,
        chunk: RetrievedChunk,
        context: RetrievalContext,
        runtime: RetrievalRuntime,
    ) -> list[RetrievedChunk | None]:
        """One optional replacement per plugin, in registration order."""
        return await self._fan(
            lambda p: p.on_retrieval_finalize(chunk, context, runtime)
        )

    async def retrieval_completed(
        self,
        chunks: Sequence[RetrievedChunk],
        context: RetrievalContext,
        runtime: RetrievalRuntime,
    ) -> None:
        await self._fan(lambda p: p.on_retrieval_completed(chunks, context, runtime))

    # --- fan-out ---

    async def _fan(self, action) -> list[Any]:
        if not self._plugins:
            return []

        async def call(plugin: Plugin) -> Any:
            # Attribute this task's `report_state` calls to the plugin. Each
            # gathered task gets its own context, so concurrent plugins don't
            # cross-talk.
            token = current_plugin.set(plugin.name)
            try:
                return await action(plugin)
            finally:
                current_plugin.reset(token)

        return list(await asyncio.gather(*(call(plugin) for plugin in self._plugins)))
