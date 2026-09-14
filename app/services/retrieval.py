from app.llm.base import LLMProvider
from app.models.chunk import RetrievedChunk
from app.plugin.context import RetrievalContext
from app.plugin.registry import PluginRegistry
from app.plugin.runtime import RetrievalRuntime
from app.retrievers.base import Retriever


class RetrievalService:
    """Retrieves chunks and finalizes them through the plugins' lifecycle methods."""

    def __init__(
        self,
        *,
        retriever: Retriever,
        llm: LLMProvider,
        registry: PluginRegistry,
    ) -> None:
        self._retriever = retriever
        self._llm = llm
        self._registry = registry

    async def retrieve(
        self,
        user_query: str,
        where: dict | None = None,
    ) -> list[RetrievedChunk]:
        """Retrieves chunks given a user query.

        `where` is a simple `{column: value}` condition map the retrievers
        apply at the index (for example `{"chat_id": ...}` to bound the
        retrieval to one chat), so the top_k budget is spent on the matching
        chunks instead of being filtered away afterwards.
        """

        context = RetrievalContext(user_query=user_query)

        runtime = RetrievalRuntime(
            context=context,
            llm=self._llm,
        )

        chunks = await self._retriever.retrieve(user_query, where)

        finalized: list[RetrievedChunk] = []

        for chunk in chunks:
            replacements = await self._registry.retrieval_finalize(
                chunk, context, runtime
            )

            replacement = next(
                (result for result in reversed(replacements) if result is not None),
                None,
            )

            finalized.append(replacement if replacement is not None else chunk)

        await self._registry.retrieval_completed(finalized, context, runtime)

        return finalized
