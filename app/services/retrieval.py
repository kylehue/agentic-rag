from collections.abc import Sequence

from app.llm.base import LLMProvider
from app.models.chunk import RetrievedChunk
from app.plugin.context import RetrievalContext
from app.plugin.registry import PluginRegistry
from app.plugin.runtime import RetrievalRuntime
from app.rerankers.base import Reranker
from app.retrievers.base import Retriever


class RetrievalService:
    """Retrieves chunks, optionally re-ranks them by relevance, and finalizes
    them through the plugins' lifecycle methods."""

    def __init__(
        self,
        *,
        retriever: Retriever,
        llm: LLMProvider,
        registry: PluginRegistry,
        reranker: Reranker | None = None,
        top_k: int = 5,
    ) -> None:
        self._retriever = retriever
        self._llm = llm
        self._registry = registry
        self._reranker = reranker
        self._top_k = top_k

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

        # Re-score and reorder the candidate set by relevance to the query.
        # A no-op when no reranker is configured.
        if self._reranker is not None:
            chunks = await self._reranker.rerank(user_query, chunks)

        # The retrievers fetch a wide candidate pool; keep the final top_k,
        # at most one chunk per dedup key, so one logical unit (e.g. an image
        # and its description) cannot occupy two of the slots.
        chunks = self._top_diverse(chunks, self._top_k)

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

    @staticmethod
    def _top_diverse(
        chunks: Sequence[RetrievedChunk], top_k: int
    ) -> list[RetrievedChunk]:
        """The top_k chunks, at most one per non-null dedup key.

        Chunks arrive best-first, so the first chunk seen for a key is its
        best. A chunk whose key an earlier (better) chunk already used is
        dropped; chunks with a null key never dedup against anything.
        """
        seen: set[str] = set()
        result: list[RetrievedChunk] = []
        for chunk in chunks:
            if len(result) >= top_k:
                break
            if chunk.key is not None:
                if chunk.key in seen:
                    continue
                seen.add(chunk.key)
            result.append(chunk)
        return result
