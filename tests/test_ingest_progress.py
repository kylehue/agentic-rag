import asyncio
import json

from app.ingest.events import JobEvents, ProgressEmitter
from app.retrievers.base import Retriever
from app.ingest.queue import IngestQueue
from app.models.ingest import (
    INGEST_DONE,
    INGEST_ERROR,
    IngestEvent,
)
from app.plugin.context import IngestionFile
from app.plugin.registry import PluginRegistry
from app.plugins.table import TablePlugin
from app.services.ingestion import IngestionService
from app.services.rag import RagService
from app.store_file.local import LocalFileStorage
from app.store_sql.local import LocalSqlStorage

from fakes import FakeEmbedder, FakeLLM, FakeVectorStorage

ANALYSIS = {
    "workbook_description": "Regional sales figures.",
    "tables": [{"index": 0, "description": "Sales per region."}],
}
CSV_BYTES = b"region,amount\nnorth,10\nsouth,25\n"


class NoopRetriever(Retriever):
    async def retrieve(self, user_query, where=None):
        return []


def make_file(name="sales.csv", content=CSV_BYTES, ctype="text/csv"):
    return IngestionFile(filename=name, content_type=ctype, file_bytes=content)


# --- JobEvents bus ---


def test_job_events_replay_then_live_tail():
    events = JobEvents()
    events.publish(IngestEvent(name="queued", payload={"file": "a", "position": 0}))
    events.publish(IngestEvent(name="started", payload={"file": "a"}))

    async def flow():
        received = []

        async def sub():
            async for event in events.subscribe():
                received.append(event.name)

        task = asyncio.create_task(sub())
        await asyncio.sleep(0)  # let the subscriber register
        events.publish(IngestEvent(name="file_done", payload={"file": "a"}))
        events.publish(IngestEvent(name="done", payload={"files": ["a"]}))
        await asyncio.wait_for(task, timeout=1)
        return received

    # Replays the two buffered events, then tails the two live ones, ending at done.
    assert asyncio.run(flow()) == ["queued", "started", "file_done", "done"]


def test_job_events_finished_before_subscribe_replays_all():
    events = JobEvents()
    events.publish(IngestEvent(name="chat", payload={"chat_id": "c"}))
    events.publish(IngestEvent(name="done", payload={"files": [], "total_chunks": 0}))

    async def flow():
        return [event.name async for event in events.subscribe()]

    assert asyncio.run(flow()) == ["chat", "done"]


def test_job_events_error_is_terminal():
    events = JobEvents()
    events.publish(IngestEvent(name="started", payload={"file": "a"}))
    events.publish(IngestEvent(name="error", payload={"error": "boom"}))

    async def flow():
        return [event.name async for event in events.subscribe()]

    assert asyncio.run(flow()) == ["started", "error"]
    assert events.finished


# --- IngestQueue ---


def test_ingest_queue_processes_a_job_and_records_events():
    processed = []

    async def process(job):
        processed.append(job.job_id)
        job.events.publish(
            IngestEvent(name=INGEST_DONE, payload={"files": [], "total_chunks": 1})
        )

    queue = IngestQueue(process_job=process, workers=1)

    async def flow():
        await queue.start()
        job = queue.enqueue("chat-1", [make_file()])
        await queue.join()
        await queue.stop()
        return job

    job = asyncio.run(flow())
    assert processed == [job.job_id]
    assert job.status == "done"
    names = [event.name for event in job.events.events]
    # The chat frame leads, the queued frame follows, and done terminates.
    assert names[0] == "chat"
    assert "queued" in names
    assert names[-1] == "done"


def test_ingest_queue_reports_an_error_when_processing_fails():
    async def process(job):
        raise RuntimeError("boom")

    queue = IngestQueue(process_job=process, workers=1)

    async def flow():
        await queue.start()
        job = queue.enqueue("chat-1", [make_file()])
        await queue.join()
        await queue.stop()
        return job

    job = asyncio.run(flow())
    assert job.status == "error"
    names = [event.name for event in job.events.events]
    assert INGEST_ERROR in names
    assert any(
        event.name == INGEST_ERROR and event.payload.get("error") == "boom"
        for event in job.events.events
    )


# --- pipeline + plugin events ---


def _build_service(tmp_path, plugins):
    sql_storage = LocalSqlStorage(storage_dir=tmp_path / "sql")
    file_storage = LocalFileStorage(storage_dir=tmp_path / "file")
    service = IngestionService(
        registry=PluginRegistry(),
        llm=FakeLLM(json.dumps(ANALYSIS)),
        embedder=FakeEmbedder(),
        vector_storage=FakeVectorStorage(),
        sql_storage=sql_storage,
        file_storage=file_storage,
    )
    for plugin in plugins:
        service._registry.register(plugin)
    return service, sql_storage


def test_ingest_emits_pipeline_and_plugin_events(tmp_path):
    service, sql_storage = _build_service(tmp_path, [TablePlugin()])
    events = JobEvents()
    emitter = ProgressEmitter(events)

    async def flow():
        await service.initialize()
        await service.ingest(make_file(), chat_id="chat-1", emitter=emitter)
        await sql_storage.close()
        return list(events.events)

    emitted = asyncio.run(flow())
    names = [event.name for event in emitted]
    assert "stage" in names and "plugin_state" in names

    stages = {event.payload["stage"] for event in emitted if event.name == "stage"}
    assert {"processing", "saving", "embedding"} <= stages
    # The pipeline stages are attributed to the ingested file.
    assert all(
        event.payload["file"] == "sales.csv" for event in emitted if event.name == "stage"
    )

    states = {
        (event.payload["plugin"], event.payload["state"])
        for event in emitted
        if event.name == "plugin_state"
    }
    assert ("table", "reading_tables") in states
    assert ("table", "analyzing_tables") in states


def test_ingest_without_emitter_emits_nothing(tmp_path):
    service, sql_storage = _build_service(tmp_path, [TablePlugin()])

    async def flow():
        await service.initialize()
        return await service.ingest(make_file(), chat_id="chat-1")

    chunks = asyncio.run(flow())
    # Still ingests correctly with no progress reporting.
    assert len(chunks) == 1
    assert chunks[0].plugin == "table"


# --- RagService end-to-end (enqueue -> worker -> events) ---


def test_rag_service_enqueue_and_process_emits_full_progress(tmp_path):
    from app.services.rag import RagService

    sql_storage = LocalSqlStorage(storage_dir=tmp_path / "sql")
    file_storage = LocalFileStorage(storage_dir=tmp_path / "file")
    rag = RagService(
        llm=FakeLLM(json.dumps(ANALYSIS)),
        embedder=FakeEmbedder(),
        retriever=NoopRetriever(),
        vector_storage=FakeVectorStorage(),
        sql_storage=sql_storage,
        file_storage=file_storage,
        plugins=[TablePlugin()],
    )

    async def flow():
        await rag.initialize()
        await rag.start_ingest_queue()
        job = rag.enqueue_ingest([make_file()], chat_id="chat-1")
        await rag._ingest_queue.join()
        await rag.stop_ingest_queue()
        await sql_storage.close()
        return job

    job = asyncio.run(flow())
    names = [event.name for event in job.events.events]
    # chat -> queued -> started -> stages -> file_done -> done
    assert names[0] == "chat"
    assert "queued" in names
    assert "started" in names
    assert "file_done" in names
    assert names[-1] == "done"
    assert job.status == "done"
    done = next(event for event in job.events.events if event.name == "done")
    assert done.payload["files"] == ["sales.csv"]
    assert done.payload["total_chunks"] == 1
    # The chunk actually landed in the store.
    assert rag.get_ingest_job(job.job_id) is job


def test_list_ingest_jobs_returns_the_chats_jobs(tmp_path):
    rag = RagService(
        llm=FakeLLM(),
        embedder=FakeEmbedder(),
        retriever=NoopRetriever(),
        vector_storage=FakeVectorStorage(),
        sql_storage=LocalSqlStorage(storage_dir=tmp_path / "sql"),
        file_storage=LocalFileStorage(storage_dir=tmp_path / "file"),
    )
    # The worker pool is not started, so enqueued jobs stay "queued".
    job_a = rag.enqueue_ingest(
        [make_file(name="a.txt", content=b"x", ctype="text/plain")], "chat-1"
    )
    job_b = rag.enqueue_ingest(
        [make_file(name="b.txt", content=b"y", ctype="text/plain")], "chat-2"
    )

    jobs = rag.list_ingest_jobs("chat-1")

    # Only chat-1's job, with its status, files, and buffered progress.
    assert [job.job_id for job in jobs] == [job_a.job_id]
    assert jobs[0].status == "queued"
    assert [f.filename for f in jobs[0].files] == ["a.txt"]
    assert [event.name for event in jobs[0].events.events] == ["chat", "queued"]
    # A finished chat's list is empty when it has no jobs.
    assert rag.list_ingest_jobs("chat-3") == []
    assert all(job.job_id != job_b.job_id for job in jobs)
