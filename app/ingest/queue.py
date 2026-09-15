from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field

from app.models.ingest import INGEST_CHAT, INGEST_ERROR, IngestEvent
from app.plugin.context import IngestionFile

from .events import JobEvents, ProgressEmitter


@dataclass
class IngestJob:
    """One ingest request: a chat plus the files to ingest into it.

    The job owns its progress events (`events`); the queue tracks its status.
    Jobs and their buffered events are in-memory (lost on restart), which is
    acceptable for a one-shot progress UX; the ingested data itself is durable.
    """

    job_id: str
    chat_id: str
    files: list[IngestionFile]
    events: JobEvents = field(default_factory=JobEvents)
    status: str = "queued"  # queued, running, done, error
    created_at: float = field(default_factory=time.time)


class IngestQueue:
    """An in-process queue that runs ingest jobs on a small worker pool.

    `process_job` is the callback that actually ingests a job (the service
    layer supplies it); the queue only owns ordering, concurrency, and the
    job registry. Decoupled from ingestion itself.
    """

    def __init__(
        self,
        *,
        process_job: Callable[[IngestJob], Awaitable[None]],
        workers: int = 1,
    ) -> None:
        self._queue: asyncio.Queue[str] = asyncio.Queue()
        self._jobs: dict[str, IngestJob] = {}
        self._process_job = process_job
        self._workers = max(1, workers)
        self._tasks: list[asyncio.Task] = []

    def enqueue(self, chat_id: str, files: Sequence[IngestionFile]) -> IngestJob:
        """Register a job and queue it. Emits the `chat` and `queued` events
        synchronously so an immediate subscriber sees the full start."""
        job = IngestJob(
            job_id=uuid.uuid4().hex,
            chat_id=chat_id,
            files=list(files),
        )
        self._jobs[job.job_id] = job
        self._queue.put_nowait(job.job_id)
        job.events.publish(IngestEvent(name=INGEST_CHAT, payload={"chat_id": chat_id}))
        emitter = ProgressEmitter(job.events)
        for position, file in enumerate(job.files):
            emitter.queued(file.filename, position)
        return job

    def get(self, job_id: str) -> IngestJob | None:
        return self._jobs.get(job_id)

    def jobs_for_chat(self, chat_id: str) -> list[IngestJob]:
        return [job for job in self._jobs.values() if job.chat_id == chat_id]

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
                    job.events.publish(
                        IngestEvent(name=INGEST_ERROR, payload={"error": str(exc)})
                    )
            finally:
                self._queue.task_done()
