from abc import ABC, abstractmethod
from collections.abc import Mapping

from app.models.chunk import IngestedChunk, RetrievedChunk
from app.plugin.context import IngestionContext
from app.plugin.hooks import HookHandler
from app.plugin.runtime import FinalizeRuntime, IngestionRuntime


class Plugin(ABC):
    """Pluggable handler for one or more document types.

    A plugin decides for itself which files it wants to manage (`accepts`)
    and bundles both sides of them: ingestion (`process`) and retrieval
    (`finalize`). Plugins declare the hooks they want to tap and persist
    their own chunks through the provided runtime.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Stable plugin identifier."""

    @property
    def uses_elements(self) -> bool:
        """Whether the plugin needs unstructured-parsed elements in the context."""
        return False

    @property
    def hooks(self) -> Mapping[str, HookHandler]:
        """Hooks this plugin taps, mapped to async handlers."""
        return {}

    @abstractmethod
    def accepts(self, context: IngestionContext) -> bool:
        """Whether this plugin wants to process this document."""

    @abstractmethod
    async def process(
        self,
        context: IngestionContext,
        runtime: IngestionRuntime,
    ) -> list[IngestedChunk]:
        """Turn the source document into chunks.

        The plugin must persist the chunks it returns through the runtime.
        """

    async def finalize(
        self,
        query: str,
        chunk: RetrievedChunk,
        runtime: FinalizeRuntime,
    ) -> RetrievedChunk:
        """Refine a retrieved chunk before it is used as evidence.

        Default: no-op.
        """
        return chunk
