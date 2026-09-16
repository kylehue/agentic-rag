from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from app.utils.events import Event, EventBus


@dataclass
class Job:
    """A unit of background work: an id, a status, an event bus, and a payload.

    `group` is an optional shared key (e.g. a chat id) used to find related
    jobs; `payload` is the work itself, interpreted by the processor. The job
    owns its progress events (`events`); the queue tracks its status.
    """

    job_id: str
    events: EventBus
    payload: Any
    group: str | None = None
    status: str = "queued"  # queued, running, done, error
    created_at: float = field(default_factory=time.time)


class JobQueue:
    """An in-process job queue: an asyncio queue + a worker pool + a registry.

    `process_job` is the callback that does a job's work (supplied by the
    domain layer); the queue only owns ordering, concurrency, the registry,
    and lifecycle status transitions. It publishes no domain events itself,
    except the terminal `error_event` (built by the supplied factory) when a
    job raises -- so every job's event stream is guaranteed to end.
    """

    def __init__(
        self,
        *,
        process_job: Callable[[Job], Awaitable[None]],
        events: Callable[[], EventBus],
        workers: int = 1,
        error_event: Callable[[Exception], Event] | None = None,
    ) -> None:
        self._queue: asyncio.Queue[str] = asyncio.Queue()
        self._jobs: dict[str, Job] = {}
        self._process_job = process_job
        self._events = events
        self._workers = max(1, workers)
        self._error_event = error_event
        self._tasks: list[asyncio.Task] = []

    def enqueue(self, payload: Any, group: str | None = None) -> Job:
        """Register a job (with a fresh event bus) and queue it."""
        job = Job(
            job_id=uuid.uuid4().hex,
            events=self._events(),
            payload=payload,
            group=group,
        )
        self._jobs[job.job_id] = job
        self._queue.put_nowait(job.job_id)
        return job

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def jobs(self, group: str | None = None) -> list[Job]:
        """All registered jobs, or just those sharing `group` when given."""
        if group is None:
            return list(self._jobs.values())
        return [job for job in self._jobs.values() if job.group == group]

    async def start(self) -> None:
        if self._tasks:
            return
        self._tasks = [
            asyncio.create_task(self._worker()) for _ in range(self._workers)
        ]

    async def join(self) -> None:
        """Wait until every enqueued job has been processed (tests)."""
        await self._queue.join()

    async def stop(self) -> None:
        for task in self._tasks:
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks = []

    async def _worker(self) -> None:
        while True:
            job_id = await self._queue.get()
            try:
                job = self._jobs.get(job_id)
                if job is None:
                    continue
                job.status = "running"
                try:
                    await self._process_job(job)
                    if job.status == "running":
                        job.status = "done"
                except Exception as exc:
                    job.status = "error"
                    if self._error_event is not None and not job.events.finished:
                        job.events.publish(self._error_event(exc))
            finally:
                self._queue.task_done()
