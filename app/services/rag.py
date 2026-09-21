from collections.abc import Sequence

from app.core.config import settings
from app.embedders.base import Embedder
from app.ingest.events import ProgressEmitter
from app.llm.base import LLMProvider
from app.models.chunk import IngestedChunk, RetrievedChunk
from app.models.ingest import (
    INGEST_CHAT,
    INGEST_ERROR,
    INGEST_TERMINAL,
)
from app.plugin.base import Plugin
from app.plugin.context import IngestionFile
from app.plugin.registry import PluginRegistry
from app.rerankers.base import Reranker
from app.retrievers.base import Retriever
from app.services.ingestion import IngestionService
from app.services.retrieval import RetrievalService
from app.store_file.base import FileStorage
from app.store_sql.base import SqlStorage
from app.store_vector.base import VectorStorage
from app.utils.events import Event, EventBus
from app.utils.queue import Job, JobQueue


class RagService:
    def __init__(
        self,
        *,
        llm: LLMProvider,
        embedder: Embedder,
        retriever: Retriever,
        vector_storage: VectorStorage,
        sql_storage: SqlStorage,
        file_storage: FileStorage,
        plugins: Sequence[Plugin] | None = None,
        reranker: Reranker | None = None,
        retrieval_top_k: int = 5,
    ) -> None:
        registry = PluginRegistry()
        for plugin in (plugins if plugins is not None else []):
            registry.register(plugin)

        self._llm = llm
        self._embedder = embedder
        self._retriever = retriever
        self._reranker = reranker
        self._vector_storage = vector_storage
        self._sql_storage = sql_storage
        self._file_storage = file_storage
        self._ingestion_service = IngestionService(
            registry=registry,
            llm=llm,
            embedder=embedder,
            vector_storage=vector_storage,
            sql_storage=sql_storage,
            file_storage=file_storage,
        )
        self._retrieval_service = RetrievalService(
            retriever=retriever,
            llm=llm,
            registry=registry,
            reranker=reranker,
            top_k=retrieval_top_k,
        )
        # Background ingest queue: ingest requests are enqueued here and run
        # on a worker pool, reporting progress through each job's events. The
        # queue is the generic app/utils JobQueue; the ingest-specific bits
        # (the event names, the per-job bus terminal set, the error event) are
        # supplied here.
        self._ingest_queue = JobQueue(
            process_job=self._process_ingest_job,
            events=lambda: EventBus(terminal=INGEST_TERMINAL),
            workers=settings.INGEST_WORKERS,
            error_event=lambda exc: Event(name=INGEST_ERROR, payload={"error": str(exc)}),
        )

    @property
    def llm(self) -> LLMProvider:
        return self._llm

    @property
    def embedder(self) -> Embedder:
        return self._embedder

    @property
    def retriever(self) -> Retriever:
        return self._retriever

    @property
    def reranker(self) -> Reranker | None:
        return self._reranker

    @property
    def vector_storage(self) -> VectorStorage:
        return self._vector_storage

    @property
    def sql_storage(self) -> SqlStorage:
        return self._sql_storage

    @property
    def file_storage(self) -> FileStorage:
        return self._file_storage

    async def ingest(
        self,
        file_bytes: bytes,
        filename: str,
        content_type: str,
        description: str | None = None,
        chat_id: str | None = None,
    ) -> tuple[str, list[IngestedChunk]]:
        """Store and index a file into `chat_id`; return its origin source
        id and chunks."""
        file = IngestionFile(
            filename=filename,
            content_type=content_type,
            file_bytes=file_bytes,
            description=description,
        )
        chunks = await self._ingestion_service.ingest(file, chat_id)
        return file.source_id, chunks

    # --- background ingest (queue + progress) ---

    async def start_ingest_queue(self) -> None:
        """Start the ingest worker pool (called at application startup)."""
        await self._ingest_queue.start()

    async def stop_ingest_queue(self) -> None:
        """Stop the ingest worker pool (called at application shutdown)."""
        await self._ingest_queue.stop()

    def enqueue_ingest(self, files: Sequence[IngestionFile], chat_id: str) -> list[Job]:
        """Queue each file for background ingestion into `chat_id` as its own
        job, and return the jobs (each id addresses its own progress stream).

        One job per file is what lets the worker pool ingest a single upload
        batch in parallel (bounded by `INGEST_WORKERS`), instead of one job
        processing its files one at a time.
        """
        jobs: list[Job] = []
        for file in files:
            job = self._ingest_queue.enqueue(payload=file, group=chat_id)
            # Ingest-specific start events (the generic queue emits none).
            job.events.publish(Event(name=INGEST_CHAT, payload={"chat_id": chat_id}))
            ProgressEmitter(job.events).queued(file.filename)
            jobs.append(job)
        return jobs

    def get_ingest_job(self, job_id: str) -> Job | None:
        """A queued/running/finished ingest job, or None."""
        return self._ingest_queue.get(job_id)

    def list_ingest_jobs(self, chat_id: str) -> list[Job]:
        """The ingest jobs for a chat (queued, running, and finished), each
        carrying its buffered progress events."""
        return self._ingest_queue.jobs(group=chat_id)

    async def _process_ingest_job(self, job: Job) -> None:
        """The worker's job: ingest the job's single file (reporting progress)
        then mark the job done."""
        file = job.payload
        emitter = ProgressEmitter(job.events)
        emitter.started(file.filename)
        chunks = await self._ingestion_service.ingest(file, job.group, emitter)
        emitter.file_done(file.filename, len(chunks))
        emitter.done(file.filename, len(chunks))

    async def retrieve(
        self,
        user_query: str,
        chat_id: str | None = None,
    ) -> list[RetrievedChunk]:
        """Return evidence without asking the agent.

        `chat_id` bounds the corpus to that chat's chunks (mapped to the
        retrievers' `where` condition); None searches the whole store.
        """
        where = {"chat_id": chat_id} if chat_id is not None else None
        return await self._retrieval_service.retrieve(user_query, where)

    async def delete_file(
        self, origin_source_id: str, chat_id: str | None = None
    ) -> int:
        """Reverse ingestion for one file (its whole emission tree), scoped to
        the chat. Returns the number of files removed (0 = not in the chat)."""
        return await self._ingestion_service.delete_file(origin_source_id, chat_id)

    async def delete_chat(self, chat_id: str) -> None:
        """Reverse ingestion for a whole chat: its files, chunks, and vectors."""
        await self._ingestion_service.delete_chat(chat_id)

    async def get_file_metadata(self, source_id: str) -> dict | None:
        """The document record for one stored file, or None if absent."""
        return await self._ingestion_service.get_file_metadata(source_id)

    async def get_file_link(self, source_id: str) -> str | None:
        """The URL at which one stored file can be retrieved, or None."""
        return await self._ingestion_service.get_file_link(source_id)

    async def list_files(self, chat_id: str | None = None) -> list[dict]:
        """The origin files (user uploads) in the chat, excluding emitted files."""
        return await self._ingestion_service.list_files(chat_id)

    async def list_file_chunks(
        self, origin_source_id: str, chat_id: str | None = None
    ) -> list[dict]:
        """All chunks of a file's emission tree (origin + emitted descendants)."""
        return await self._ingestion_service.list_file_chunks(origin_source_id, chat_id)
