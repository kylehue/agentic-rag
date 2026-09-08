import asyncio
import json
import logging
from pathlib import Path
from unittest.mock import patch

import pytest

from app.core.config import settings
from app.errors.document import InvalidDocumentError
from app.plugin.base import Plugin
from app.plugin.hooks import (
    HookBus,
    IngestionProcessPayload,
    hook,
)
from app.plugin.registry import PluginRegistry
from app.plugins import text as text_module
from app.services.ingestion import IngestionService
from app.store_file.local import LocalFileStorage
from app.store_sql.local import LocalSqlStorage

from fakes import (
    FakeEmbedder,
    FakeLLM,
    FakeVectorStorage,
    make_ingestion_file,
    make_table_element,
    make_text_element,
)
from app.plugins.table import TablePlugin
from app.plugins.text import TextPlugin

ANALYSIS = {
    "workbook_description": "Regional sales figures.",
    "tables": [
        {
            "index": 0,
            "description": "Sales per region.",
            "role": "fact data",
            "schema": [
                {"name": "region", "description": "sales region"},
                {"name": "amount", "description": "total amount"},
            ],
            "relationships": [],
        }
    ],
}


class SidecarPlugin(Plugin):
    """A plugin that accepts every CSV and generates one marker chunk."""

    @property
    def name(self) -> str:
        return "sidecar"

    def accepts(self, context) -> bool:
        return context.file.filename.endswith(".csv")

    @hook("ingestion_process")
    async def _process(
        self,
        payload: IngestionProcessPayload,
    ) -> list:
        from app.models.chunk import IngestedChunk

        if not self.accepts(payload["context"]):
            return []

        return [IngestedChunk(plugin=self.name, text="sidecar marker")]


class EmptyPlugin(Plugin):
    """A plugin that accepts everything but generates no chunks."""

    @property
    def name(self) -> str:
        return "empty"

    def accepts(self, context) -> bool:
        return True

    @hook("ingestion_process")
    async def _process(self, payload: IngestionProcessPayload) -> list:
        return []


class UnacceptedEmitterPlugin(Plugin):
    """Accepts .txt files, chunks them, and emits a file no plugin accepts."""

    @property
    def name(self) -> str:
        return "unaccepted-emitter"

    def accepts(self, context) -> bool:
        return context.file.filename.endswith(".txt")

    @hook("ingestion_process")
    async def _process(
        self,
        payload: IngestionProcessPayload,
    ) -> list:
        from app.models.chunk import IngestedChunk

        runtime = payload["runtime"]
        await runtime.emit_file(
            filename="data.unknown",
            content_type="application/x-unknown",
            file_bytes=b"mystery bytes",
        )
        return [IngestedChunk(plugin=self.name, text="txt body")]


class FailsOnEmitPlugin(Plugin):
    """Accepts .txt and .csv. Processing a .txt emits a .csv; processing the
    .csv raises, simulating a single process failing part-way through."""

    @property
    def name(self) -> str:
        return "fails-on-emit"

    def accepts(self, context) -> bool:
        return context.file.filename.endswith((".txt", ".csv"))

    @hook("ingestion_process")
    async def _process(
        self,
        payload: IngestionProcessPayload,
    ) -> list:
        from app.models.chunk import IngestedChunk

        context = payload["context"]
        runtime = payload["runtime"]

        if context.file.filename.endswith(".csv"):
            raise RuntimeError("simulated plugin failure on the emitted file")

        await runtime.emit_file(
            filename="embedded.csv",
            content_type="text/csv",
            file_bytes=b"a,b\n1,2",
        )
        return [IngestedChunk(plugin=self.name, text="txt body")]


class ChainEmitterPlugin(Plugin):
    """Accepts ``step<N>.*`` files, chunks them, and emits ``step<N+1>`` up to
    N=2. Used to build a 3-deep emission chain (grandchild lineage)."""

    @property
    def name(self) -> str:
        return "chain"

    def accepts(self, context) -> bool:
        return context.file.filename.startswith("step")

    @hook("ingestion_process")
    async def _process(
        self,
        payload: IngestionProcessPayload,
    ) -> list:
        from app.models.chunk import IngestedChunk

        context = payload["context"]
        runtime = payload["runtime"]

        level = int(context.file.filename.split("step")[1].split(".")[0])

        if level < 2:
            await runtime.emit_file(
                filename=f"step{level + 1}.csv",
                content_type="text/csv",
                file_bytes=b"a,b\n1,2",
            )

        return [IngestedChunk(plugin=self.name, text=f"level {level}")]


def build_service(tmp_path, *, llm=None, hooks=None, plugins=None):
    llm = llm if llm is not None else FakeLLM()
    sql_storage = LocalSqlStorage(storage_dir=tmp_path / "sql")
    file_storage = LocalFileStorage(storage_dir=tmp_path / "file")
    vector_storage = FakeVectorStorage()
    hooks = hooks if hooks is not None else HookBus()
    registry = PluginRegistry(hooks)
    for plugin in (plugins if plugins is not None else [TextPlugin(), TablePlugin()]):
        registry.register(plugin)

    service = IngestionService(
        hooks=hooks,
        registry=registry,
        llm=llm,
        embedder=FakeEmbedder(),
        vector_storage=vector_storage,
        sql_storage=sql_storage,
        file_storage=file_storage,
    )

    return service, sql_storage, file_storage, vector_storage, hooks


def test_csv_ingest_end_to_end(tmp_path):
    llm = FakeLLM(json.dumps(ANALYSIS))
    service, sql_storage, file_storage, vector_storage, _ = build_service(
        tmp_path,
        llm=llm,
    )

    async def flow():
        await service.initialize()
        chunks = await service.ingest(
            make_ingestion_file(
                file_bytes=b"region,amount\nnorth,10\n",
                filename="sales.csv",
                content_type="text/csv",
            )
        )
        documents = await sql_storage.get_all(settings.DOCUMENT_METADATA_TABLE_NAME)
        chunk_rows = await sql_storage.get_all(settings.CHUNK_TABLE_NAME)
        db_tables = {
            row["name"]
            for row in await sql_storage.query(
                "SELECT name FROM sqlite_master WHERE type='table'",
                limit=None,
            )
        }
        await sql_storage.close()
        return chunks, documents, chunk_rows, db_tables

    chunks, documents, chunk_rows, db_tables = asyncio.run(flow())

    # One table chunk.
    assert len(chunks) == 1
    assert chunks[0].plugin == "table"

    # The source file was stored and recorded as an origin document.
    assert len(documents) == 1
    assert documents[0]["file_orig_filename"] == "sales.csv"
    assert documents[0]["is_origin"]
    assert Path(documents[0]["file_path"]).is_file()

    # Chunk record was persisted by the service, metadata as-is, without SQL
    # rows or a stored chunk file.
    assert len(chunk_rows) == 1
    row = chunk_rows[0]
    assert row["source_id"] is not None
    # Lineage lives in columns: a top-level file has no parent, and its
    # origin is itself.
    assert row["parent_source_id"] is None
    assert row["origin_source_id"] == row["source_id"]
    metadata = row["metadata"]
    assert metadata["table_name"] == "sales"
    assert metadata["schema"]
    assert "sql_rows" not in metadata
    assert "parent_source_id" not in metadata
    assert "origin_source_id" not in metadata
    assert "file_path" not in metadata

    # No per-table data tables were created; only the system tables exist.
    assert db_tables == {
        settings.CHUNK_TABLE_NAME,
        settings.DOCUMENT_METADATA_TABLE_NAME,
    }

    # The vector was added.
    assert vector_storage.added[0][0] == [chunks[0].chunk_id]


def test_text_ingest_emits_and_subprocesses_embedded_table(tmp_path):
    # A text document with an embedded table yields text chunks, and the
    # table is emitted as a CSV (with a description of the surrounding
    # context) that the TablePlugin sub-processes into one chunk.
    llm = FakeLLM(json.dumps(ANALYSIS))
    service, sql_storage, file_storage, vector_storage, hooks = build_service(
        tmp_path,
        llm=llm,
    )

    emitted_events: list[dict] = []
    subprocessed_events: list[dict] = []

    async def on_emitted(payload):
        emitted_events.append(payload)

    async def on_subprocessed(payload):
        subprocessed_events.append(payload)

    hooks.register("file_emitted", on_emitted)
    hooks.register("file_subprocessed", on_subprocessed)

    table_html = (
        "<table><thead><tr><th>region</th><th>amount</th></tr></thead>"
        "<tbody><tr><td>north</td><td>10</td></tr></tbody></table>"
    )
    elements = [
        make_text_element("Quarterly report body.", page_number=1),
        make_table_element(table_html, page_number=1),
    ]

    async def flow():
        await service.initialize()
        with patch.object(
            text_module,
            "partition",
            return_value=elements,
        ):
            chunks = await service.ingest(
                make_ingestion_file(
                    file_bytes=b"fake document bytes",
                    filename="report.pdf",
                    content_type="application/pdf",
                )
            )
        documents = await sql_storage.get_all(settings.DOCUMENT_METADATA_TABLE_NAME)
        chunk_rows = await sql_storage.get_all(settings.CHUNK_TABLE_NAME)
        await sql_storage.close()
        return chunks, documents, chunk_rows

    chunks, documents, chunk_rows = asyncio.run(flow())

    # Every file that went through ingestion is stored, exactly once: the
    # original upload (is_origin) and the emitted CSV (not is_origin).
    assert len(documents) == 2
    original = next(
        d for d in documents if d["file_orig_filename"] == "report.pdf"
    )
    emitted_doc = next(
        d for d in documents if d["file_orig_filename"] == "report_table_1.csv"
    )
    assert original["is_origin"]
    assert not emitted_doc["is_origin"]
    assert Path(emitted_doc["file_path"]).is_file()
    # No duplicate storage: two files, two distinct paths.
    assert original["file_path"] != emitted_doc["file_path"]

    # Emission and subprocess both happened exactly once.
    assert len(emitted_events) == 1
    emitted_file = emitted_events[0]["emitted_file"]
    assert emitted_file.filename == "report_table_1.csv"
    assert emitted_file.content_type == "text/csv"
    # The emitted description carries the text found around the table.
    assert emitted_file.description is not None
    assert "Quarterly report body." in emitted_file.description
    assert len(subprocessed_events) == 1
    assert len(subprocessed_events[0]["chunks"]) == 1

    # One text chunk (parent) and one table chunk (the sub-processed CSV).
    text_chunks = [c for c in chunks if c.plugin == "text"]
    table_chunks = [c for c in chunks if c.plugin == "table"]
    assert len(text_chunks) == 1
    assert "Quarterly report body." in text_chunks[0].text
    assert len(table_chunks) == 1
    assert table_chunks[0].metadata["table_name"] == "report_table_1"

    assert len(chunk_rows) == 2

    # The table chunk carries lineage back to the emitting document, in
    # columns (not metadata).
    table_row = next(r for r in chunk_rows if r["plugin"] == "table")
    assert table_row["source_id"] != original["source_id"]
    assert table_row["source_id"] == emitted_file.source_id
    assert table_row["parent_source_id"] == original["source_id"]
    assert table_row["origin_source_id"] == original["source_id"]
    assert "parent_source_id" not in table_row["metadata"]
    assert "origin_source_id" not in table_row["metadata"]
    assert "file_path" not in table_row["metadata"]

    # The text chunk is top-level: no parent, origin is itself.
    text_row = next(r for r in chunk_rows if r["plugin"] == "text")
    assert text_row["source_id"] == original["source_id"]
    assert text_row["parent_source_id"] is None
    assert text_row["origin_source_id"] == original["source_id"]
    assert "parent_source_id" not in text_row["metadata"]

    # The description reached the TablePlugin's LLM as source context.
    assert any("Quarterly report body." in prompt for prompt in llm.prompts)


def test_file_description_reaches_subprocess_llm(tmp_path):
    # A caller-provided file description flows into the context, into the
    # emitted file's description, and into the TablePlugin's LLM prompt.
    llm = FakeLLM(json.dumps(ANALYSIS))
    service, sql_storage, *_ = build_service(tmp_path, llm=llm)

    table_html = (
        "<table><thead><tr><th>region</th><th>amount</th></tr></thead>"
        "<tbody><tr><td>north</td><td>10</td></tr></tbody></table>"
    )
    elements = [make_table_element(table_html, page_number=1)]

    async def flow():
        await service.initialize()
        with patch.object(
            text_module,
            "partition",
            return_value=elements,
        ):
            await service.ingest(
                make_ingestion_file(
                    file_bytes=b"fake document bytes",
                    filename="report.pdf",
                    content_type="application/pdf",
                    description="Acme Q3 sales pack, internal use only.",
                )
            )
        await sql_storage.close()

    asyncio.run(flow())

    assert any(
        "Acme Q3 sales pack, internal use only." in prompt for prompt in llm.prompts
    )


def test_multiple_accepting_plugins_all_run(tmp_path):
    llm = FakeLLM(json.dumps(ANALYSIS))
    service, sql_storage, *_ = build_service(
        tmp_path,
        llm=llm,
        plugins=[TextPlugin(), TablePlugin(), SidecarPlugin()],
    )

    async def flow():
        await service.initialize()
        chunks = await service.ingest(
            make_ingestion_file(
                file_bytes=b"region,amount\nnorth,10\n",
                filename="sales.csv",
                content_type="text/csv",
            )
        )
        chunk_rows = await sql_storage.get_all(settings.CHUNK_TABLE_NAME)
        await sql_storage.close()
        return chunks, chunk_rows

    chunks, chunk_rows = asyncio.run(flow())

    # TablePlugin and SidecarPlugin both accepted the CSV and both ran.
    table_chunks = [c for c in chunks if c.plugin == "table"]
    sidecar_chunks = [c for c in chunks if c.text == "sidecar marker"]
    assert len(table_chunks) == 1
    assert len(sidecar_chunks) == 1
    assert len(chunk_rows) == 2

    # TextPlugin rejected the CSV: the only non-table chunk is the
    # sidecar's, not a text chunk.
    non_table_rows = [row for row in chunk_rows if row["plugin"] != "table"]
    assert len(non_table_rows) == 1
    assert non_table_rows[0]["plugin"] == "sidecar"
    assert non_table_rows[0]["text"] == "sidecar marker"


def test_ingest_without_chunks_warns(tmp_path, caplog):
    service, sql_storage, *_ = build_service(tmp_path, plugins=[EmptyPlugin()])

    async def flow():
        await service.initialize()
        with caplog.at_level(logging.WARNING, logger="app.services.ingestion"):
            chunks = await service.ingest(
                make_ingestion_file(
                    file_bytes=b"some content",
                    filename="anything.txt",
                    content_type="text/plain",
                )
            )
        await sql_storage.close()
        return chunks

    chunks = asyncio.run(flow())

    assert chunks == []
    assert any("produced no chunks" in record.message for record in caplog.records)


def test_file_no_plugin_accepts_raises(tmp_path):
    service, sql_storage, *_ = build_service(tmp_path)

    async def flow():
        await service.initialize()
        with pytest.raises(InvalidDocumentError):
            await service.ingest(
                make_ingestion_file(
                    file_bytes=b"x",
                    filename="photo.png",
                    content_type="image/png",
                )
            )
        await sql_storage.close()

    asyncio.run(flow())


def test_file_rejected_by_all_registered_plugins_raises(tmp_path):
    service, sql_storage, *_ = build_service(
        tmp_path,
        plugins=[TextPlugin()],
    )

    async def flow():
        await service.initialize()
        with pytest.raises(InvalidDocumentError):
            await service.ingest(
                make_ingestion_file(
                    file_bytes=b"a,b\n1,2",
                    filename="data.csv",
                    content_type="text/csv",
                )
            )
        await sql_storage.close()

    asyncio.run(flow())


def test_origin_propagates_through_emission_chain(tmp_path):
    # A 3-deep emission chain: step0 -> step1 -> step2. Every level's chunk
    # must point its parent_source_id at its direct emitter and its
    # origin_source_id at the top-most file (step0), not just the parent.
    service, sql_storage, *_ = build_service(
        tmp_path,
        plugins=[ChainEmitterPlugin()],
    )

    async def flow():
        await service.initialize()
        await service.ingest(
            make_ingestion_file(
                file_bytes=b"root",
                filename="step0.txt",
                content_type="text/plain",
                source_id="root-id",
            )
        )
        documents = await sql_storage.get_all(settings.DOCUMENT_METADATA_TABLE_NAME)
        chunk_rows = await sql_storage.get_all(settings.CHUNK_TABLE_NAME)
        await sql_storage.close()
        return documents, chunk_rows

    documents, chunk_rows = asyncio.run(flow())

    # Every file in the chain is stored exactly once; only the top-level
    # upload is marked as origin.
    stored = {d["file_orig_filename"]: d for d in documents}
    assert set(stored) == {"step0.txt", "step1.csv", "step2.csv"}
    assert stored["step0.txt"]["is_origin"]
    assert not stored["step1.csv"]["is_origin"]
    assert not stored["step2.csv"]["is_origin"]

    row_by_level = {r["text"]: r for r in chunk_rows}
    step0, step1, step2 = (
        row_by_level[f"level {i}"]["source_id"] for i in range(3)
    )

    # Level 0 (top): no parent, origin is itself.
    assert step0 == "root-id"
    assert row_by_level["level 0"]["parent_source_id"] is None
    assert row_by_level["level 0"]["origin_source_id"] == step0

    # Level 1 (child): parent is step0, origin is step0.
    assert step1 != step0
    assert row_by_level["level 1"]["parent_source_id"] == step0
    assert row_by_level["level 1"]["origin_source_id"] == step0

    # Level 2 (grandchild): parent is step1, but origin is still step0.
    assert step2 not in (step0, step1)
    assert row_by_level["level 2"]["parent_source_id"] == step1
    assert row_by_level["level 2"]["origin_source_id"] == step0


def test_unaccepted_emitted_file_is_ignored_with_warning(tmp_path, caplog):
    # An emitted file that no plugin accepts is not stored anywhere and is
    # skipped with a warning; the parent ingestion completes normally.
    service, sql_storage, *_ = build_service(
        tmp_path,
        plugins=[UnacceptedEmitterPlugin()],
    )

    async def flow():
        await service.initialize()
        with caplog.at_level(logging.WARNING, logger="app.services.ingestion"):
            chunks = await service.ingest(
                make_ingestion_file(
                    file_bytes=b"txt bytes",
                    filename="notes.txt",
                    content_type="text/plain",
                )
            )
        documents = await sql_storage.get_all(settings.DOCUMENT_METADATA_TABLE_NAME)
        chunk_rows = await sql_storage.get_all(settings.CHUNK_TABLE_NAME)
        await sql_storage.close()
        return chunks, documents, chunk_rows

    chunks, documents, chunk_rows = asyncio.run(flow())

    # The parent's chunk is still produced and stored.
    assert len(chunks) == 1
    assert chunks[0].text == "txt body"
    assert len(chunk_rows) == 1

    # Only the original file is a source document; the emitted file was
    # ignored (no document row, no chunks).
    assert [d["file_orig_filename"] for d in documents] == ["notes.txt"]
    assert all(r["plugin"] == "unaccepted-emitter" for r in chunk_rows)

    # The ignore was logged.
    assert any(
        "weren't accepted by any plugins" in record.message for record in caplog.records
    )


def test_file_completed_per_file_and_ingestion_completed_once_at_the_end(tmp_path):
    # file_completed fires for every file the pipeline walks (children first,
    # the emitting parent last) and nothing is committed while it runs.
    # ingestion_completed fires exactly once, at the very end, after the
    # commit.
    llm = FakeLLM(json.dumps(ANALYSIS))
    service, sql_storage, _files, _vectors, hooks = build_service(
        tmp_path,
        llm=llm,
    )

    file_completed_events: list[dict] = []
    ingestion_completed_events: list[dict] = []
    documents_at_file_completed: list[dict] = []
    documents_at_ingestion_completed: list[dict] = []
    chunk_rows_at_ingestion_completed: list[dict] = []

    async def on_file_completed(payload):
        file_completed_events.append(payload)
        documents_at_file_completed.extend(
            await sql_storage.get_all(settings.DOCUMENT_METADATA_TABLE_NAME)
        )

    async def on_ingestion_completed(payload):
        ingestion_completed_events.append(payload)
        documents_at_ingestion_completed.extend(
            await sql_storage.get_all(settings.DOCUMENT_METADATA_TABLE_NAME)
        )
        chunk_rows_at_ingestion_completed.extend(
            await sql_storage.get_all(settings.CHUNK_TABLE_NAME)
        )

    hooks.register("file_completed", on_file_completed)
    hooks.register("ingestion_completed", on_ingestion_completed)

    table_html = (
        "<table><thead><tr><th>region</th><th>amount</th></tr></thead>"
        "<tbody><tr><td>north</td><td>10</td></tr></tbody></table>"
    )
    elements = [
        make_text_element("Quarterly report body.", page_number=1),
        make_table_element(table_html, page_number=1),
    ]

    async def flow():
        await service.initialize()
        with patch.object(
            text_module,
            "partition",
            return_value=elements,
        ):
            chunks = await service.ingest(
                make_ingestion_file(
                    file_bytes=b"fake document bytes",
                    filename="report.pdf",
                    content_type="application/pdf",
                )
            )
        await sql_storage.close()
        return chunks

    chunks = asyncio.run(flow())

    # file_completed: once per file, emitted child before the parent.
    assert [e["context"].file.filename for e in file_completed_events] == [
        "report_table_1.csv",
        "report.pdf",
    ]
    child_event, parent_event = file_completed_events
    assert child_event["context"].parent_file is not None
    assert parent_event["context"].parent_file is None
    # The parent's payload carries the whole subtree: its own chunk plus
    # the sub-processed table chunk.
    assert [c.plugin for c in parent_event["chunks"]] == ["text", "table"]

    # Nothing is committed while file_completed is running.
    assert documents_at_file_completed == []

    # ingestion_completed: exactly once, at the very end, after the commit.
    assert len(ingestion_completed_events) == 1
    final = ingestion_completed_events[0]
    assert final["context"].file.filename == "report.pdf"
    assert final["context"].parent_file is None
    assert final["chunks"] == chunks
    assert final["runtime"] is parent_event["runtime"]
    assert len(documents_at_ingestion_completed) == 2
    assert len(chunk_rows_at_ingestion_completed) == 2


def test_process_failure_saves_nothing(tmp_path):
    # If any single process in the emission tree fails, nothing is saved:
    # the source document, chunk records, and vectors are all collected during
    # the walk and only committed at the very end.
    llm = FakeLLM()
    service, sql_storage, file_storage, vector_storage, _ = build_service(
        tmp_path,
        llm=llm,
        plugins=[FailsOnEmitPlugin()],
    )

    async def flow():
        await service.initialize()
        with pytest.raises(RuntimeError):
            await service.ingest(
                make_ingestion_file(
                    file_bytes=b"txt",
                    filename="report.txt",
                    content_type="text/plain",
                )
            )
        documents = await sql_storage.get_all(settings.DOCUMENT_METADATA_TABLE_NAME)
        chunk_rows = await sql_storage.get_all(settings.CHUNK_TABLE_NAME)
        await sql_storage.close()
        return documents, chunk_rows

    documents, chunk_rows = asyncio.run(flow())

    # The failure propagated out of ingest.
    assert documents == []
    assert chunk_rows == []
    assert vector_storage.added == []

    # No files were written to disk (documents/).
    storage_dir = Path(tmp_path) / "file"
    written = (
        [p for p in storage_dir.rglob("*") if p.is_file()]
        if storage_dir.exists()
        else []
    )
    assert written == []
