import asyncio
from dataclasses import replace

import pytest
from pydantic import ValidationError

from app.api_schemas.chunk import RetrievedChunkSchema
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
        origin_source_id="s1",
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


def test_from_dict_requires_origin_source_id():
    data = {
        "chunk_id": "c1",
        "source_id": "s1",
        "origin_source_id": "s1",
        "plugin": "text",
        "text": "t",
        "parent_source_id": None,
    }

    chunk = RetrievedChunk.from_dict(data, score=0.4)
    assert chunk.origin_source_id == "s1"
    assert chunk.parent_source_id is None

    without_origin = {k: v for k, v in data.items() if k != "origin_source_id"}
    with pytest.raises(KeyError):
        RetrievedChunk.from_dict(without_origin)


def test_retrieved_chunk_schema_rejects_missing_origin():
    chunk = make_chunk("text")

    # A well-formed chunk (origin always set) validates.
    schema = RetrievedChunkSchema.model_validate(chunk)
    assert schema.origin_source_id == "s1"

    # A chunk without an origin is invalid.
    broken = replace(chunk, origin_source_id=None)  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        RetrievedChunkSchema.model_validate(broken)


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
