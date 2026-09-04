import asyncio
from dataclasses import replace

import pytest

from app.models.chunk import RetrievedChunk
from app.plugin.base import Plugin
from app.plugin.hooks import HookBus, HOOK_CHUNK_FINALIZED
from app.plugin.registry import PluginRegistry
from app.plugin.runtime import FinalizeRuntime

from fakes import FakeLLM, make_context


class NoopPlugin(Plugin):
    def __init__(self, name: str, accepts_all: bool = True) -> None:
        self._name = name
        self._accepts_all = accepts_all

    @property
    def name(self) -> str:
        return self._name

    def accepts(self, context) -> bool:
        return self._accepts_all

    async def process(self, context, runtime):
        return []


def test_accepting_plugins_returns_willing_plugins_in_order():
    registry = PluginRegistry(HookBus())
    first = NoopPlugin("first")
    second = NoopPlugin("second", accepts_all=False)
    third = NoopPlugin("third")
    for plugin in (first, second, third):
        registry.register(plugin)

    accepting = registry.accepting_plugins(make_context(filename="doc.pdf"))

    assert accepting == [first, third]


def test_duplicate_plugin_name_rejected():
    registry = PluginRegistry(HookBus())
    registry.register(NoopPlugin("a"))

    with pytest.raises(ValueError):
        registry.register(NoopPlugin("a"))


def test_plugin_hooks_are_wired_on_register():
    hooks = HookBus()
    registry = PluginRegistry(hooks)
    events: list[str] = []

    class TappingPlugin(NoopPlugin):
        @property
        def hooks(self):
            async def on_started(context):
                events.append(str(context))

            return {"ingestion.started": on_started}

    registry.register(TappingPlugin("t"))

    asyncio.run(hooks.emit("ingestion.started", context="ctx-1"))

    assert events == ["ctx-1"]


def make_chunk(plugin: str = "table") -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id="chunk-1",
        source_id="source-1",
        plugin=plugin,
        text="original text",
        score=0.9,
    )


def test_finalize_routes_to_producing_plugin():
    hooks = HookBus()
    registry = PluginRegistry(hooks)
    finalized_events: list[RetrievedChunk] = []

    class MarkingPlugin(NoopPlugin):
        async def finalize(self, query, chunk, runtime):
            return replace(chunk, text=f"{chunk.text} [finalized]")

    registry.register(MarkingPlugin("m"))

    async def on_finalized(query, chunk):
        finalized_events.append(chunk)

    hooks.register(HOOK_CHUNK_FINALIZED, on_finalized)

    runtime = FinalizeRuntime(llm=FakeLLM(), hooks=hooks)

    result = asyncio.run(registry.finalize("q", make_chunk("m"), runtime))

    assert result.text == "original text [finalized]"
    assert len(finalized_events) == 1
    assert finalized_events[0].text == "original text [finalized]"


def test_finalize_unknown_plugin_is_passthrough():
    hooks = HookBus()
    registry = PluginRegistry(hooks)
    registry.register(NoopPlugin("t"))
    emitted: list[dict] = []

    async def on_finalized(**payload):
        emitted.append(payload)

    hooks.register(HOOK_CHUNK_FINALIZED, on_finalized)

    runtime = FinalizeRuntime(llm=FakeLLM(), hooks=hooks)
    chunk = make_chunk("unknown")

    result = asyncio.run(registry.finalize("q", chunk, runtime))

    assert result is chunk
    assert emitted == []
