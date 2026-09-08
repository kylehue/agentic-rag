import asyncio

import pytest

from app.plugin.base import Plugin
from app.plugin.hooks import HookBus, IngestionStartedPayload, hook
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


def emit_started(hooks: HookBus, context) -> None:
    parts = build_runtime(context, hooks=hooks)
    asyncio.run(
        hooks.trigger(
            "ingestion_started",
            IngestionStartedPayload(context=context, runtime=parts.runtime),
        )
    )


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


def test_decorated_hook_handlers_are_wired_on_register():
    hooks = HookBus()
    registry = PluginRegistry(hooks)
    events: list = []

    class TappingPlugin(NoopPlugin):
        @hook("ingestion_started")
        async def on_started(self, payload: IngestionStartedPayload) -> None:
            events.append(payload["context"])

    context = make_context()
    registry.register(TappingPlugin("t"))
    emit_started(hooks, context)

    assert events == [context]


def test_decorated_handlers_receieve_runtime():
    hooks = HookBus()
    registry = PluginRegistry(hooks)
    seen: list = []

    class RuntimeSpyPlugin(NoopPlugin):

        @hook("ingestion_started")
        async def on_started(self, payload: IngestionStartedPayload) -> None:
            seen.append(payload["runtime"])

    registry.register(RuntimeSpyPlugin("spy"))
    context = make_context()
    emit_started(hooks, context)

    assert len(seen) == 1
    assert seen[0].context is context


def test_plugin_for_returns_registered_plugin():
    registry = PluginRegistry(HookBus())
    plugin = NoopPlugin("a")
    registry.register(plugin)

    assert registry.plugin_for("a") is plugin
    assert registry.plugin_for("missing") is None
