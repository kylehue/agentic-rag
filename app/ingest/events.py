from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

from app.models.ingest import (
    INGEST_DONE,
    INGEST_ERROR,
    INGEST_FILE_DONE,
    INGEST_PLUGIN_STATE,
    INGEST_QUEUED,
    INGEST_STARTED,
    INGEST_STAGE,
    IngestEvent,
)

_SENTINEL = object()


class JobEvents:
    """One job's progress events: a replayable buffer plus live subscribers.

    `publish` is synchronous (no await), so in the single-threaded event loop
    a subscriber registers atomically with respect to publishing: it replays
    exactly the events buffered before it subscribed, then tails the live
    ones, with no gaps and no duplicates. It is in-memory and per-process,
    which matches the app's local-storage, single-instance shape; a
    multi-instance deployment would back this with a broker.
    """

    def __init__(self) -> None:
        self._events: list[IngestEvent] = []
        self._subscribers: list[asyncio.Queue] = []
        self._finished = False

    @property
    def finished(self) -> bool:
        return self._finished

    @property
    def events(self) -> list[IngestEvent]:
        """The buffered events (for replay and inspection)."""
        return list(self._events)

    def publish(self, event: IngestEvent) -> None:
        self._events.append(event)
        for queue in self._subscribers:
            queue.put_nowait(event)
        if event.name in (INGEST_DONE, INGEST_ERROR):
            self._finished = True
            for queue in self._subscribers:
                queue.put_nowait(_SENTINEL)

    def _register(self) -> tuple[asyncio.Queue, int]:
        # Synchronous on purpose: atomic with respect to `publish`.
        queue: asyncio.Queue = asyncio.Queue()
        self._subscribers.append(queue)
        return queue, len(self._events)

    async def subscribe(self) -> AsyncIterator[IngestEvent]:
        queue, start = self._register()
        try:
            for event in self._events[:start]:
                yield event
            if self._finished:
                return
            while True:
                item = await queue.get()
                if item is _SENTINEL:
                    break
                yield item
        finally:
            self._subscribers.remove(queue)


class ProgressEmitter:
    """The publishing side of a job's events, with per-event-name helpers.

    The pipeline and plugins hold one of these (or None when running without
    progress reporting) and call its methods; they never touch the bus or the
    event model directly.
    """

    def __init__(self, events: JobEvents) -> None:
        self._events = events

    def _publish(self, name: str, **payload: Any) -> None:
        self._events.publish(IngestEvent(name=name, payload=payload))

    def queued(self, file: str, position: int) -> None:
        self._publish(INGEST_QUEUED, file=file, position=position)

    def started(self, file: str) -> None:
        self._publish(INGEST_STARTED, file=file)

    def stage(self, file: str, stage: str, **detail: Any) -> None:
        self._publish(INGEST_STAGE, file=file, stage=stage, **detail)

    def plugin_state(self, file: str, plugin: str | None, state: str, **detail: Any) -> None:
        self._publish(INGEST_PLUGIN_STATE, file=file, plugin=plugin, state=state, **detail)

    def file_done(self, file: str, chunk_count: int) -> None:
        self._publish(INGEST_FILE_DONE, file=file, chunk_count=chunk_count)

    def done(self, files: list[str], total_chunks: int) -> None:
        self._publish(INGEST_DONE, files=files, total_chunks=total_chunks)

    def error(self, file: str | None, error: str) -> None:
        self._publish(INGEST_ERROR, file=file, error=error)
