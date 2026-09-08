import asyncio

from app.models.chunk import IngestedChunk
from app.plugin.hooks import (
    HookBus,
    IngestionCompletedPayload,
    IngestionProcessPayload,
)

from fakes import build_runtime, make_context


def test_emit_without_handlers_returns_empty():
    bus = HookBus()
    context = make_context()
    parts = build_runtime(context)
    payload = IngestionCompletedPayload(
        context=context,
        chunks=[],
        runtime=parts.runtime,
    )
    assert asyncio.run(bus.trigger("ingestion_completed", payload)) == []


def test_emit_passes_payload_to_handler():
    bus = HookBus()
    seen: list = []

    async def handler(payload):
        seen.append(payload)

    bus.register("ingestion_completed", handler)

    context = make_context()
    parts = build_runtime(context)
    chunk = IngestedChunk(plugin="text", text="c1")
    payload = IngestionCompletedPayload(
        context=context,
        chunks=[chunk],
        runtime=parts.runtime,
    )
    asyncio.run(bus.trigger("ingestion_completed", payload))

    assert seen == [payload]
    assert seen[0]["context"] is context
    assert seen[0]["chunks"] == [chunk]


def test_emit_returns_handler_results_in_registration_order():
    bus = HookBus()

    async def first(payload):
        return ["chunk-a"]

    async def second(payload):
        return ["chunk-b", "chunk-c"]

    bus.register("ingestion_process", first)
    bus.register("ingestion_process", second)

    context = make_context()
    parts = build_runtime(context)
    payload = IngestionProcessPayload(context=context, runtime=parts.runtime)

    results = asyncio.run(bus.trigger("ingestion_process", payload))

    assert results == [["chunk-a"], ["chunk-b", "chunk-c"]]


def test_unregister_removes_handler():
    bus = HookBus()
    events: list = []

    async def handler(payload):
        events.append("hit")

    bus.register("file_emitted", handler)
    bus.unregister("file_emitted", handler)

    context = make_context()
    parts = build_runtime(context)
    asyncio.run(
        bus.trigger(
            "file_emitted",
            {"context": context, "emitted_file": object(), "runtime": parts.runtime},
        )
    )

    assert events == []


def test_handlers_lists_registered_handlers():
    bus = HookBus()

    async def handler(payload):
        pass

    bus.register("ingestion_completed", handler)

    assert bus.handlers("ingestion_completed") == [handler]
    assert bus.handlers("unknown.hook") == []
