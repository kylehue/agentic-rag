import asyncio
from dataclasses import replace

import pytest

from app.models.chunk import IngestedTextChunk, RetrievedTextChunk
from app.plugin.base import Plugin
from app.plugin.context import RetrievalContext
from app.plugin.registry import PluginRegistry

from fakes import build_runtime, make_context


class NoopPlugin(Plugin):
    def __init__(self, name: str, accepts_all: bool = True) -> None:
        self._name = name
        self._accepts_all = accepts_all

    @property
    def name(self) -> str:
        return self._name

    def accepts(self, context) -> bool:
        return self._accepts_all


def test_accepting_plugins_returns_willing_plugins_in_order():
    registry = PluginRegistry()
    first = NoopPlugin("first")
    second = NoopPlugin("second", accepts_all=False)
    third = NoopPlugin("third")
    for plugin in (first, second, third):
        registry.register(plugin)

    accepting = registry.accepting_plugins(make_context(filename="doc.pdf"))

    assert accepting == [first, third]


def test_duplicate_plugin_name_rejected():
    registry = PluginRegistry()
    registry.register(NoopPlugin("a"))

    with pytest.raises(ValueError):
        registry.register(NoopPlugin("a"))


def test_plugin_for_returns_registered_plugin():
    registry = PluginRegistry()
    plugin = NoopPlugin("a")
    registry.register(plugin)

    assert registry.plugin_for("a") is plugin
    assert registry.plugin_for("missing") is None


def test_events_reach_every_registered_plugin_with_context_and_runtime():
    registry = PluginRegistry()
    events: list = []

    class TappingPlugin(NoopPlugin):
        async def on_ingestion_started(self, context, runtime) -> None:
            events.append((context, runtime))

    context = make_context()
    registry.register(TappingPlugin("t"))
    parts = build_runtime(context, registry=registry)
    asyncio.run(registry.ingestion_started(context, parts.runtime))

    assert events == [(context, parts.runtime)]


def test_ingestion_process_aggregates_chunks_in_registration_order():
    registry = PluginRegistry()

    class ChunkPlugin(NoopPlugin):
        def __init__(self, name: str, texts: list[str]) -> None:
            super().__init__(name)
            self._texts = texts

        async def on_ingestion_process(self, context, runtime) -> list:
            return [
                IngestedTextChunk(plugin=self.name, text=text) for text in self._texts
            ]

    registry.register(ChunkPlugin("a", ["a1"]))
    registry.register(ChunkPlugin("b", ["b1", "b2"]))
    registry.register(ChunkPlugin("c", []))

    context = make_context()
    parts = build_runtime(context, registry=registry)
    chunks = asyncio.run(registry.ingestion_process(context, parts.runtime))

    assert [chunk.text for chunk in chunks] == ["a1", "b1", "b2"]


def test_ingestion_process_skips_plugins_that_do_not_accept():
    registry = PluginRegistry()
    called: list = []

    class Plugin_(NoopPlugin):
        def __init__(self, name: str, accept: bool) -> None:
            super().__init__(name, accepts_all=accept)

        async def on_ingestion_process(self, context, runtime) -> list:
            called.append(self.name)
            return [IngestedTextChunk(plugin=self.name, text=self.name)]

    registry.register(Plugin_("ok", accept=True))
    registry.register(Plugin_("no", accept=False))

    context = make_context()
    parts = build_runtime(context, registry=registry)
    chunks = asyncio.run(registry.ingestion_process(context, parts.runtime))

    # The non-accepting plugin is never invoked, so it needs no self-gate.
    assert called == ["ok"]
    assert [chunk.text for chunk in chunks] == ["ok"]


def test_retrieval_finalize_returns_one_result_per_plugin():
    registry = PluginRegistry()
    chunk = RetrievedTextChunk(
        chunk_id="c1",
        source_id="s1",
        origin_source_id="s1",
        plugin="a",
        text="t",
        score=0.5,
    )

    class Finalizer(NoopPlugin):
        def __init__(self, name: str, replacement: str | None) -> None:
            super().__init__(name)
            self._replacement = replacement

        async def on_retrieval_finalize(self, chunk, context, runtime):
            if self._replacement is None:
                return None
            return replace(chunk, text=self._replacement)

    registry.register(Finalizer("none", None))
    registry.register(Finalizer("renamer", "renamed"))

    context = make_context()
    parts = build_runtime(context, registry=registry)
    retrieval_context = RetrievalContext(user_query="q")
    results = asyncio.run(
        registry.retrieval_finalize(chunk, retrieval_context, parts.runtime)
    )

    assert results[0] is None
    assert results[1] is not None
    assert results[1].text == "renamed"


def test_empty_registry_fans_out_to_nothing():
    registry = PluginRegistry()
    context = make_context()
    parts = build_runtime(context, registry=registry)

    assert (
        asyncio.run(registry.ingestion_process(context, parts.runtime)) == []
    )
    asyncio.run(registry.ingestion_started(context, parts.runtime))
