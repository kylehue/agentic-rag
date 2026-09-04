import asyncio
from collections.abc import Awaitable, Callable

# Hook names fired by the system. Plugins tap them by mapping a hook name to
# an async handler in `Plugin.hooks`.
HOOK_INGESTION_STARTED = "ingestion.started"  # (context)
HOOK_PLUGINS_SELECTED = "ingestion.plugins_selected"  # (context, plugins)
HOOK_CHUNK_SAVED = "chunk.saved"  # (context, chunk)
HOOK_FILE_EMITTED = "file.emitted"  # (context, emitted_file)
HOOK_FILE_SUBPROCESSED = "file.subprocessed"  # (emitted_file, chunks)
HOOK_INGESTION_COMPLETED = "ingestion.completed"  # (context, chunks)
HOOK_CHUNK_FINALIZED = "chunk.finalized"  # (query, chunk)
HOOK_RETRIEVAL_COMPLETED = "retrieval.completed"  # (query, chunks)

HookHandler = Callable[..., Awaitable[object]]


class HookBus:
    """Named event bus for system extension points.

    The system emits hooks at lifecycle points (ingestion, retrieval) and
    plugins register async handlers to tap into them. Handlers receive the
    hook payload as keyword arguments and run concurrently.
    """

    def __init__(self) -> None:
        self._handlers: dict[str, list[HookHandler]] = {}

    def register(self, name: str, handler: HookHandler) -> None:
        self._handlers.setdefault(name, []).append(handler)

    def unregister(self, name: str, handler: HookHandler) -> None:
        handlers = self._handlers.get(name)
        if handlers is not None and handler in handlers:
            handlers.remove(handler)

    def handlers(self, name: str) -> list[HookHandler]:
        return list(self._handlers.get(name, []))

    async def emit(self, name: str, **payload: object) -> list[object]:
        """Fire a hook and return the results of all handlers."""
        handlers = self._handlers.get(name)
        if not handlers:
            return []
        return list(
            await asyncio.gather(*(handler(**payload) for handler in handlers))
        )
