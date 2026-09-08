from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Literal, TypeVar, TypedDict, overload

from app.models.chunk import IngestedChunk, RetrievedChunk
from app.plugin.context import IngestionContext, IngestionFile, RetrievalContext

if TYPE_CHECKING:
    from app.plugin.runtime import IngestionRuntime, RetrievalRuntime

# --- hook names ---

INGESTION_STARTED = Literal["ingestion_started"]
INGESTION_PROCESS = Literal["ingestion_process"]
FILE_EMITTED = Literal["file_emitted"]
FILE_SUBPROCESSED = Literal["file_subprocessed"]
FILE_COMPLETED = Literal["file_completed"]
INGESTION_COMPLETED = Literal["ingestion_completed"]
RETRIEVAL_FINALIZE = Literal["retrieval_finalize"]
RETRIEVAL_COMPLETED = Literal["retrieval_completed"]

# --- hook payloads ---


class IngestionStartedPayload(TypedDict):
    context: IngestionContext
    runtime: IngestionRuntime


class IngestionProcessPayload(TypedDict):
    context: IngestionContext
    runtime: IngestionRuntime


class FileEmittedPayload(TypedDict):
    context: IngestionContext
    emitted_file: IngestionFile
    runtime: IngestionRuntime


class FileSubprocessedPayload(TypedDict):
    emitted_file: IngestionFile
    chunks: Sequence[IngestedChunk]
    runtime: IngestionRuntime


class FileCompletedPayload(TypedDict):
    """One file (and its emitted subtree) finished processing.

    Fired for every file the pipeline walks, before anything is committed.
    ``chunks`` is the whole subtree: the file's own chunks plus every
    descendant's.
    """

    context: IngestionContext
    chunks: Sequence[IngestedChunk]
    runtime: IngestionRuntime


class IngestionCompletedPayload(TypedDict):
    """The whole ingestion finished: every file processed and everything committed."""

    context: IngestionContext
    chunks: Sequence[IngestedChunk]
    runtime: IngestionRuntime


class RetrievalFinalizePayload(TypedDict):
    context: RetrievalContext
    chunk: RetrievedChunk
    runtime: RetrievalRuntime


class RetrievalCompletedPayload(TypedDict):
    context: RetrievalContext
    chunks: Sequence[RetrievedChunk]
    runtime: RetrievalRuntime


# --- hook handlers (signatures) ---

IngestionStartedHandler = Callable[[IngestionStartedPayload], Awaitable[None]]
IngestionProcessHandler = Callable[
    [IngestionProcessPayload],
    Awaitable[Sequence[IngestedChunk] | None],
]
FileEmittedHandler = Callable[[FileEmittedPayload], Awaitable[None]]
FileSubprocessedHandler = Callable[[FileSubprocessedPayload], Awaitable[None]]
FileCompletedHandler = Callable[[FileCompletedPayload], Awaitable[None]]
IngestionCompletedHandler = Callable[[IngestionCompletedPayload], Awaitable[None]]
RetrievalFinalizeHandler = Callable[
    [RetrievalFinalizePayload], Awaitable[RetrievedChunk | None]
]
RetrievalCompletedHandler = Callable[[RetrievalCompletedPayload], Awaitable[None]]

# Fallback typing for custom (non-builtin) hooks.
HookHandler = Callable[[Mapping[str, Any]], Awaitable[Any]]


# --- @hook decorator ---

_S = TypeVar("_S")
_M = TypeVar("_M", bound=Callable[..., Any])


@overload
def hook(
    name: INGESTION_STARTED,
) -> Callable[
    [Callable[[_S, IngestionStartedPayload], Awaitable[None]]],
    Callable[[_S, IngestionStartedPayload], Awaitable[None]],
]: ...


@overload
def hook(
    name: INGESTION_PROCESS,
) -> Callable[
    [
        Callable[
            [_S, IngestionProcessPayload], Awaitable[Sequence[IngestedChunk] | None]
        ]
    ],
    Callable[[_S, IngestionProcessPayload], Awaitable[Sequence[IngestedChunk] | None]],
]: ...


@overload
def hook(
    name: FILE_EMITTED,
) -> Callable[
    [Callable[[_S, FileEmittedPayload], Awaitable[None]]],
    Callable[[_S, FileEmittedPayload], Awaitable[None]],
]: ...


@overload
def hook(
    name: FILE_SUBPROCESSED,
) -> Callable[
    [Callable[[_S, FileSubprocessedPayload], Awaitable[None]]],
    Callable[[_S, FileSubprocessedPayload], Awaitable[None]],
]: ...


@overload
def hook(
    name: FILE_COMPLETED,
) -> Callable[
    [Callable[[_S, FileCompletedPayload], Awaitable[None]]],
    Callable[[_S, FileCompletedPayload], Awaitable[None]],
]: ...


@overload
def hook(
    name: INGESTION_COMPLETED,
) -> Callable[
    [Callable[[_S, IngestionCompletedPayload], Awaitable[None]]],
    Callable[[_S, IngestionCompletedPayload], Awaitable[None]],
]: ...


@overload
def hook(
    name: RETRIEVAL_FINALIZE,
) -> Callable[
    [Callable[[_S, RetrievalFinalizePayload], Awaitable[RetrievedChunk | None]]],
    Callable[[_S, RetrievalFinalizePayload], Awaitable[RetrievedChunk | None]],
]: ...


@overload
def hook(
    name: RETRIEVAL_COMPLETED,
) -> Callable[
    [Callable[[_S, RetrievalCompletedPayload], Awaitable[None]]],
    Callable[[_S, RetrievalCompletedPayload], Awaitable[None]],
]: ...


@overload
def hook(name: str) -> Callable[[_M], _M]: ...


def hook(name: str):
    """Mark a plugin method as a handler for the named hook.

    The registry discovers decorated methods on plugin classes and wires
    them into the shared HookBus when the plugin is registered. The
    handler receives the hook's typed payload as its single argument.
    """

    def decorator(method: Any, /) -> Any:
        setattr(method, "__hook_name__", name)
        return method

    return decorator


# --- hook bus ---


class HookBus:
    """Event bus that drives the pipeline.

    Services emit hooks with typed payloads; plugins react by registering
    handlers (via the `@hook` decorator). Handlers run concurrently and
    receive the payload as a single argument. Two hooks are response
    hooks whose handler return values the services consume:

    - `ingestion_process` handlers return the chunks they generated.
    - `retrieval_finalize` handlers return a replacement chunk, or None
      to leave the chunk unchanged.
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

    @overload
    async def trigger(
        self,
        name: INGESTION_STARTED,
        payload: IngestionStartedPayload,
    ) -> list[None]: ...

    @overload
    async def trigger(
        self,
        name: INGESTION_PROCESS,
        payload: IngestionProcessPayload,
    ) -> list[Sequence[IngestedChunk] | None]: ...

    @overload
    async def trigger(
        self,
        name: FILE_EMITTED,
        payload: FileEmittedPayload,
    ) -> list[None]: ...

    @overload
    async def trigger(
        self,
        name: FILE_SUBPROCESSED,
        payload: FileSubprocessedPayload,
    ) -> list[None]: ...

    @overload
    async def trigger(
        self,
        name: FILE_COMPLETED,
        payload: FileCompletedPayload,
    ) -> list[None]: ...

    @overload
    async def trigger(
        self,
        name: INGESTION_COMPLETED,
        payload: IngestionCompletedPayload,
    ) -> list[None]: ...

    @overload
    async def trigger(
        self,
        name: RETRIEVAL_FINALIZE,
        payload: RetrievalFinalizePayload,
    ) -> list[RetrievedChunk | None]: ...

    @overload
    async def trigger(
        self,
        name: RETRIEVAL_COMPLETED,
        payload: RetrievalCompletedPayload,
    ) -> list[None]: ...

    @overload
    async def trigger(self, name: str, payload: Mapping[str, Any]) -> list[Any]: ...

    async def trigger(self, name: str, payload: Mapping[str, Any]) -> list[Any]:
        """Fire a hook and return the results of all handlers."""
        handlers = self._handlers.get(name)
        if not handlers:
            return []
        return list(await asyncio.gather(*(handler(payload) for handler in handlers)))
