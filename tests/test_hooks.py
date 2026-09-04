import asyncio

from app.plugin.hooks import HookBus, HOOK_CHUNK_SAVED, HOOK_FILE_EMITTED


def test_emit_without_handlers_returns_empty():
    bus = HookBus()
    assert asyncio.run(bus.emit(HOOK_CHUNK_SAVED, chunk="x")) == []


def test_emit_passes_payload_to_handler():
    bus = HookBus()
    seen: dict = {}

    async def handler(context=None, chunk=None):
        seen["context"] = context
        seen["chunk"] = chunk

    bus.register(HOOK_CHUNK_SAVED, handler)

    asyncio.run(bus.emit(HOOK_CHUNK_SAVED, context="ctx", chunk="chunk-1"))

    assert seen == {"context": "ctx", "chunk": "chunk-1"}


def test_emit_runs_all_handlers():
    bus = HookBus()
    events: list[str] = []

    async def first(**payload):
        events.append("first")

    async def second(**payload):
        events.append("second")

    bus.register(HOOK_FILE_EMITTED, first)
    bus.register(HOOK_FILE_EMITTED, second)

    asyncio.run(bus.emit(HOOK_FILE_EMITTED, emitted_file="f"))

    assert set(events) == {"first", "second"}


def test_unregister_removes_handler():
    bus = HookBus()
    events: list[str] = []

    async def handler(**payload):
        events.append("hit")

    bus.register(HOOK_CHUNK_SAVED, handler)
    bus.unregister(HOOK_CHUNK_SAVED, handler)

    asyncio.run(bus.emit(HOOK_CHUNK_SAVED, chunk="x"))

    assert events == []


def test_handlers_lists_registered_handlers():
    bus = HookBus()

    async def handler(**payload):
        pass

    bus.register(HOOK_CHUNK_SAVED, handler)

    assert bus.handlers(HOOK_CHUNK_SAVED) == [handler]
    assert bus.handlers("unknown.hook") == []
