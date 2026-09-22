from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING

from app.models.chunk import IngestedChunk, RetrievedChunk
from app.plugin.context import IngestionContext, IngestionFile, RetrievalContext

if TYPE_CHECKING:
    from app.plugin.runtime import IngestionRuntime, RetrievalRuntime


class Plugin(ABC):
    """Pluggable document handler.

    A plugin is identity (``name``), file selection (``accepts``), and a set
    of overridable lifecycle methods. The registry fans every pipeline event
    out to every registered plugin -- concurrently, in registration order --
    and each plugin decides for itself whether to act (typically by guarding
    with ``accepts`` or by recognizing its own chunks). All lifecycle methods
    default to no-ops, so a plugin overrides only what it cares about.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Stable plugin identifier. Stamped on the chunks this plugin produces."""

    @abstractmethod
    def accepts(self, context: IngestionContext) -> bool:
        """Whether this plugin wants to process this document."""

    # --- ingestion lifecycle ---
    #
    # Event-specific data comes first; ``context`` and ``runtime`` are always
    # the last two arguments.

    async def on_ingestion_started(
        self,
        context: IngestionContext,
        runtime: IngestionRuntime,
    ) -> None:
        """A file is about to be processed."""

    async def on_ingestion_process(
        self,
        context: IngestionContext,
        runtime: IngestionRuntime,
    ) -> Sequence[IngestedChunk]:
        """Generate the chunks for this file. Return [] to contribute nothing."""
        return []

    async def on_file_emitted(
        self,
        emitted_file: IngestionFile,
        context: IngestionContext,
        runtime: IngestionRuntime,
    ) -> None:
        """The run for ``context`` emitted a file for subprocess."""

    async def on_file_subprocessed(
        self,
        emitted_file: IngestionFile,
        chunks: Sequence[IngestedChunk],
        context: IngestionContext,
        runtime: IngestionRuntime,
    ) -> None:
        """An emitted file finished processing; ``chunks`` is its subtree."""

    async def on_file_completed(
        self,
        chunks: Sequence[IngestedChunk],
        context: IngestionContext,
        runtime: IngestionRuntime,
    ) -> None:
        """A file (and its emitted subtree) finished processing, pre-commit."""

    async def on_ingestion_completed(
        self,
        chunks: Sequence[IngestedChunk],
        context: IngestionContext,
        runtime: IngestionRuntime,
    ) -> None:
        """The whole ingestion finished, after everything was committed."""

    # --- retrieval lifecycle ---

    async def on_retrieval_finalize(
        self,
        chunk: RetrievedChunk,
        context: RetrievalContext,
        runtime: RetrievalRuntime,
    ) -> RetrievedChunk | None:
        """Optionally replace a retrieved chunk. Return None to leave it."""
        return None

    async def on_retrieval_completed(
        self,
        chunks: Sequence[RetrievedChunk],
        context: RetrievalContext,
        runtime: RetrievalRuntime,
    ) -> None:
        """Retrieval finished; ``chunks`` is the finalized result set."""
