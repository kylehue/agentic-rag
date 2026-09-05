import asyncio
from dataclasses import replace

from app.models.chunk import RetrievedChunk
from app.plugin.base import Plugin
from app.plugin.hooks import HookBus, RetrievalFinalizePayload, hook
from app.plugin.registry import PluginRegistry
from app.retrievers.base import Retriever
from app.services.retrieval import RetrievalService

from fakes import FakeLLM


class FakeRetriever(Retriever):
    def __init__(self, chunks: list[RetrievedChunk]) -> None:
        self._chunks = chunks

    async def retrieve(self, user_query):
        return self._chunks


class EnrichingPlugin(Plugin):
    """Finalizes only the chunks it produced, via the retrieval.finalize hook."""

    def __init__(self, name: str) -> None:
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def accepts(self, context) -> bool:
        return False

    @hook("retrieval_finalize")
    async def on_finalize(self, payload: RetrievalFinalizePayload):
        chunk = payload["chunk"]
        if chunk.plugin != self._name:
            return None
        return replace(
            chunk, text=f"{chunk.text} [enriched with {payload['context'].user_query}]"
        )


def make_chunk(plugin: str, chunk_id: str = "c1") -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        source_id="s1",
        plugin=plugin,
        text=f"text from {plugin}",
        score=0.5,
    )


def build_service(hooks, retriever):
    return RetrievalService(
        retriever=retriever,
        llm=FakeLLM(),
        hooks=hooks,
    )


def test_finalize_hook_enriches_only_owning_chunks():
    hooks = HookBus()
    registry = PluginRegistry(hooks)
    registry.register(EnrichingPlugin("enricher"))

    service = build_service(
        hooks,
        FakeRetriever([make_chunk("enricher", "c1"), make_chunk("text", "c2")]),
    )

    results = asyncio.run(service.retrieve("q"))

    assert results[0].text == "text from enricher [enriched with q]"
    assert results[1].text == "text from text"


def test_chunk_without_finalizing_plugin_passes_through():
    hooks = HookBus()
    service = build_service(hooks, FakeRetriever([make_chunk("unknown")]))

    results = asyncio.run(service.retrieve("q"))

    assert results[0].text == "text from unknown"


def test_retrieval_completed_hook_fires_with_finalized_chunks():
    hooks = HookBus()
    events: list[dict] = []

    async def on_completed(payload):
        events.append(payload)

    hooks.register("retrieval_completed", on_completed)

    service = build_service(hooks, FakeRetriever([make_chunk("text")]))

    asyncio.run(service.retrieve("q"))

    assert len(events) == 1
    assert len(events[0]["chunks"]) == 1
    # The completed payload carries the per-run retrieval context and runtime.
    assert events[0]["context"].user_query == "q"
    assert events[0]["runtime"].context is events[0]["context"]
