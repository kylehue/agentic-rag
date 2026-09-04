import asyncio

from app.models.chunk import RetrievedChunk
from app.plugin.hooks import HookBus, HOOK_RETRIEVAL_COMPLETED
from app.plugin.registry import PluginRegistry
from app.plugin.runtime import FinalizeRuntime
from app.retrievers.base import Retriever


class RetrievalService:
    """Retrieves chunks and routes them to their plugin's finalizer."""

    def __init__(
        self,
        *,
        retriever: Retriever,
        registry: PluginRegistry,
        finalize_runtime: FinalizeRuntime,
        hooks: HookBus,
    ) -> None:
        self._retriever = retriever
        self._registry = registry
        self._finalize_runtime = finalize_runtime
        self._hooks = hooks

    async def retrieve(self, user_query: str) -> list[RetrievedChunk]:
        """Retrieves chunks given a user query."""

        chunks = await self._retriever.retrieve(user_query)

        results = await asyncio.gather(
            *(
                self._registry.finalize(
                    user_query,
                    chunk,
                    self._finalize_runtime,
                )
                for chunk in chunks
            )
        )

        await self._hooks.emit(
            HOOK_RETRIEVAL_COMPLETED,
            query=user_query,
            chunks=list(results),
        )

        return list(results)
