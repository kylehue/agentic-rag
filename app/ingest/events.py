from __future__ import annotations

from typing import Any

from app.models.ingest import (
    INGEST_DONE,
    INGEST_ERROR,
    INGEST_FILE_DONE,
    INGEST_PLUGIN_STATE,
    INGEST_QUEUED,
    INGEST_STARTED,
    INGEST_STAGE,
)
from app.utils.events import Event, EventBus


class ProgressEmitter:
    """The publishing side of a job's events, with per-event-name helpers.

    The pipeline and plugins hold one of these (or None when running without
    progress reporting) and call its methods; they never touch the bus or the
    event model directly.
    """

    def __init__(self, events: EventBus) -> None:
        self._events = events

    def _publish(self, name: str, **payload: Any) -> None:
        self._events.publish(Event(name=name, payload=payload))

    def queued(self, file: str) -> None:
        self._publish(INGEST_QUEUED, file=file)

    def started(self, file: str) -> None:
        self._publish(INGEST_STARTED, file=file)

    def stage(self, file: str, stage: str, **detail: Any) -> None:
        self._publish(INGEST_STAGE, file=file, stage=stage, **detail)

    def plugin_state(self, file: str, plugin: str | None, state: str, **detail: Any) -> None:
        self._publish(INGEST_PLUGIN_STATE, file=file, plugin=plugin, state=state, **detail)

    def file_done(self, file: str, chunk_count: int) -> None:
        self._publish(INGEST_FILE_DONE, file=file, chunk_count=chunk_count)

    def done(self, file: str, total_chunks: int) -> None:
        self._publish(INGEST_DONE, file=file, total_chunks=total_chunks)

    def error(self, file: str | None, error: str) -> None:
        self._publish(INGEST_ERROR, file=file, error=error)
