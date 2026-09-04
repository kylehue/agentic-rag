from app.models.chunk import RetrievedChunk
from app.plugin.base import Plugin
from app.plugin.context import IngestionContext
from app.plugin.hooks import HookBus, HOOK_CHUNK_FINALIZED
from app.plugin.runtime import FinalizeRuntime


class PluginRegistry:
    """Holds registered plugins and routes documents and chunks to them.

    During ingestion it offers a document to every plugin and returns the
    ones that accept it. During retrieval it routes a chunk to the
    finalizer of the plugin that produced it.
    """

    def __init__(self, hooks: HookBus) -> None:
        self._hooks = hooks
        self._plugins: list[Plugin] = []
        self._plugins_by_name: dict[str, Plugin] = {}

    def register(self, plugin: Plugin) -> None:
        existing = self._plugins_by_name.get(plugin.name)
        if existing is not None:
            raise ValueError(
                f"Plugin name '{plugin.name}' is already registered."
            )

        self._plugins.append(plugin)
        self._plugins_by_name[plugin.name] = plugin

        for hook_name, handler in plugin.hooks.items():
            self._hooks.register(hook_name, handler)

    def plugins(self) -> list[Plugin]:
        return list(self._plugins)

    def accepting_plugins(self, context: IngestionContext) -> list[Plugin]:
        """All registered plugins that want to process this document."""
        return [plugin for plugin in self._plugins if plugin.accepts(context)]

    async def finalize(
        self,
        query: str,
        chunk: RetrievedChunk,
        runtime: FinalizeRuntime,
    ) -> RetrievedChunk:
        """Route a chunk to the finalizer of the plugin that produced it."""
        plugin = self._plugins_by_name.get(chunk.plugin)
        if plugin is None:
            return chunk

        finalized = await plugin.finalize(query, chunk, runtime)
        await self._hooks.emit(
            HOOK_CHUNK_FINALIZED,
            query=query,
            chunk=finalized,
        )
        return finalized
