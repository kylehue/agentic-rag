from abc import ABC, abstractmethod

from app.plugin.context import IngestionContext


class Plugin(ABC):
    """Pluggable document handler.

    A plugin is identity plus decorated hook handlers. It decides which
    files it wants to manage (`accepts`), generates chunks by handling the
    `ingestion.process` hook, emits embedded content through the runtime,
    and enriches its own retrieved chunks by handling the
    `retrieval.finalize` hook.

    Hook handlers are declared with the `@hook(name)` decorator:

        class MyPlugin(Plugin):
            @hook("ingestion_process")
            async def on_process(self, payload: IngestionProcessPayload) -> list[IngestedChunk]:
                ...
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Stable plugin identifier. Stamped on the chunks this plugin produces."""

    @abstractmethod
    def accepts(self, context: IngestionContext) -> bool:
        """Whether this plugin wants to process this document."""
