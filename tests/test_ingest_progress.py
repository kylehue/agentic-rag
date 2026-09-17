import asyncio
import json

from app.ingest.events import ProgressEmitter
from app.models.ingest import INGEST_TERMINAL
from app.plugin.context import IngestionFile
from app.plugin.registry import PluginRegistry
from app.plugins.table import TablePlugin
from app.retrievers.base import Retriever
from app.services.ingestion import IngestionService
from app.services.rag import RagService
from app.store_file.local import LocalFileStorage
from app.store_sql.local import LocalSqlStorage
from app.utils.events import Event, EventBus
from app.utils.queue import JobQueue

from fakes import FakeEmbedder, FakeLLM, FakeVectorStorage

ANALYSIS = {
    "workbook_description": "Regional sales figures.",
    "tables": [{"index": 0, "description": "Sales per region."}],
}
CSV_BYTES = b"region,amount\nnorth,10\nsouth,25\n"
TERMINAL = frozenset({"done", "error"})


class NoopRetriever(Retriever):
    async def retrieve(self, user_query, where=None):
        return []


def make_file(name="sales.csv", content=CSV_BYTES, ctype="text/csv"):
    return IngestionFile(filename=name, content_type=ctype, file_bytes=content)


def make_queue(process, *, terminal=TERMINAL, error_name="error"):
    return JobQueue(
        process_job=process,
        events=lambda: EventBus(terminal=terminal),
        workers=1,
        error_event=lambda exc: Event(name=error_name, payload={"error": str(exc)}),
    )


# --- EventBus bus (generic) ---


def test_event_bus_replay_then_live_tail():
    events = EventBus(terminal=TERMINAL)
    events.publish(Event(name="queued", payload={"file": "a"}))
    events.publish(Event(name="started", payload={"file": "a"}))

    async def flow():
        received = []

        async def sub():
            async for event in events.subscribe():
                received.append(event.name)

        task = asyncio.create_task(sub())
        await asyncio.sleep(0)  # let the subscriber register
        events.publish(Event(name="file_done", payload={"file": "a"}))
        events.publish(Event(name="done", payload={"files": ["a"]}))
        await asyncio.wait_for(task, timeout=1)
        return received

    # Replays the two buffered events, then tails the two live ones, ending at done.
    assert asyncio.run(flow()) == ["queued", "started", "file_done", "done"]


def test_event_bus_finished_before_subscribe_replays_all():
    events = EventBus(terminal=TERMINAL)
    events.publish(Event(name="chat", payload={"chat_id": "c"}))
    events.publish(Event(name="done", payload={"files": [], "total_chunks": 0}))

    async def flow():
        return [event.name async for event in events.subscribe()]

    assert asyncio.run(flow()) == ["chat", "done"]


def test_event_bus_error_is_terminal():
    events = EventBus(terminal=TERMINAL)
    events.publish(Event(name="started", payload={"file": "a"}))
    events.publish(Event(name="error", payload={"error": "boom"}))

    async def flow():
        return [event.name async for event in events.subscribe()]

    assert asyncio.run(flow()) == ["started", "error"]
    assert events.finished


# --- JobQueue (generic) ---


def test_job_queue_processes_a_job_and_transitions_status():
    processed = []

    async def process(job):
        processed.append(job.job_id)
        job.events.publish(Event(name="done", payload={"total": 1}))

    queue = make_queue(process)

    async def flow():
        await queue.start()
        job = queue.enqueue(payload=[make_file()], group="chat-1")
        await queue.join()
        await queue.stop()
        return job

    job = asyncio.run(flow())
    assert processed == [job.job_id]
    assert job.status == "done"
    # The generic queue emits no domain events itself; only the processor's do.
    assert [event.name for event in job.events.events] == ["done"]


def test_job_queue_reports_an_error_when_processing_fails():
    async def process(job):
        raise RuntimeError("boom")

    queue = make_queue(process)

    async def flow():
        await queue.start()
        job = queue.enqueue(payload=[make_file()], group="chat-1")
        await queue.join()
        await queue.stop()
        return job

    job = asyncio.run(flow())
    assert job.status == "error"
    # The queue published the terminal error event with the exception message.
    assert any(
        event.name == "error" and event.payload.get("error") == "boom"
        for event in job.events.events
    )


def test_job_queue_groups_jobs_by_group():
    async def process(job):
        job.events.publish(Event(name="done", payload={}))

    queue = make_queue(process)

    async def flow():
        await queue.start()
        a = queue.enqueue(payload="x", group="g1")
        b = queue.enqueue(payload="y", group="g2")
        await queue.join()
        await queue.stop()
        return a, b

    a, b = asyncio.run(flow())
    assert [job.job_id for job in queue.jobs(group="g1")] == [a.job_id]
    assert [job.job_id for job in queue.jobs(group="g2")] == [b.job_id]
    assert {job.job_id for job in queue.jobs()} == {a.job_id, b.job_id}


# --- pipeline + plugin events (ingest layer) ---


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
    events = EventBus(terminal=INGEST_TERMINAL)
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
        jobs = rag.enqueue_ingest([make_file()], chat_id="chat-1")
        await rag._ingest_queue.join()
        await rag.stop_ingest_queue()
        await sql_storage.close()
        return jobs

    [job] = asyncio.run(flow())
    names = [event.name for event in job.events.events]
    # The ingest layer emits chat -> queued -> started -> stages -> file_done -> done.
    assert names[0] == "chat"
    assert "queued" in names
    assert "started" in names
    assert "file_done" in names
    assert names[-1] == "done"
    assert job.status == "done"
    done = next(event for event in job.events.events if event.name == "done")
    assert done.payload["file"] == "sales.csv"
    assert done.payload["total_chunks"] == 1
    # The chunk actually landed in the store.
    assert rag.get_ingest_job(job.job_id) is job


def test_enqueue_ingest_makes_one_job_per_file(tmp_path):
    sql_storage = LocalSqlStorage(storage_dir=tmp_path / "sql")
    rag = RagService(
        llm=FakeLLM(json.dumps(ANALYSIS)),
        embedder=FakeEmbedder(),
        retriever=NoopRetriever(),
        vector_storage=FakeVectorStorage(),
        sql_storage=sql_storage,
        file_storage=LocalFileStorage(storage_dir=tmp_path / "file"),
        plugins=[TablePlugin()],
    )

    async def flow():
        await rag.initialize()
        await rag.start_ingest_queue()
        jobs = rag.enqueue_ingest(
            [make_file(name="a.csv"), make_file(name="b.csv")], "chat-1"
        )
        await rag._ingest_queue.join()
        await rag.stop_ingest_queue()
        await sql_storage.close()
        return jobs

    jobs = asyncio.run(flow())

    # Two files in one batch become two independent jobs, one file each.
    assert [job.payload.filename for job in jobs] == ["a.csv", "b.csv"]
    assert len({job.job_id for job in jobs}) == 2
    for job in jobs:
        assert job.status == "done"
        names = [event.name for event in job.events.events]
        assert names[0] == "chat" and names.count("queued") == 1 and names[-1] == "done"
        assert job.events.finished


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
    [job_a] = rag.enqueue_ingest(
        [make_file(name="a.txt", content=b"x", ctype="text/plain")], "chat-1"
    )
    [job_b] = rag.enqueue_ingest(
        [make_file(name="b.txt", content=b"y", ctype="text/plain")], "chat-2"
    )

    jobs = rag.list_ingest_jobs("chat-1")

    # Only chat-1's job, with its status, file (payload), and buffered progress.
    assert [job.job_id for job in jobs] == [job_a.job_id]
    assert jobs[0].status == "queued"
    assert jobs[0].payload.filename == "a.txt"
    assert [event.name for event in jobs[0].events.events] == ["chat", "queued"]
    # A chat with no jobs lists empty.
    assert rag.list_ingest_jobs("chat-3") == []
    assert all(job.job_id != job_b.job_id for job in jobs)
