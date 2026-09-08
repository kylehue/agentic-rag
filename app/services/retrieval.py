from app.llm.base import LLMProvider
from app.models.chunk import RetrievedChunk
from app.plugin.context import RetrievalContext
from app.plugin.hooks import (
    HookBus,
    RetrievalCompletedPayload,
    RetrievalFinalizePayload,
)
from app.plugin.runtime import RetrievalRuntime
from app.retrievers.base import Retriever


class RetrievalService:
    """Retrieves chunks and finalizes them through the retrieval hooks."""

    def __init__(
        self,
        *,
        retriever: Retriever,
        llm: LLMProvider,
        hooks: HookBus,
    ) -> None:
        self._retriever = retriever
        self._llm = llm
        self._hooks = hooks

    async def retrieve(self, user_query: str) -> list[RetrievedChunk]:
        """Retrieves chunks given a user query."""

        context = RetrievalContext(user_query=user_query)

        runtime = RetrievalRuntime(
            context=context,
            hooks=self._hooks,
            llm=self._llm,
        )

        chunks = await self._retriever.retrieve(user_query)

        finalized: list[RetrievedChunk] = []

        for chunk in chunks:
            replacements = await self._hooks.trigger(
                "retrieval_finalize",
                RetrievalFinalizePayload(
                    context=context,
                    chunk=chunk,
                    runtime=runtime,
                ),
            )

            replacement = next(
                (result for result in reversed(replacements) if result is not None),
                None,
            )

            finalized.append(replacement if replacement is not None else chunk)

        await self._hooks.trigger(
            "retrieval_completed",
            RetrievalCompletedPayload(
                context=context,
                chunks=finalized,
                runtime=runtime,
            ),
        )

        return finalized
