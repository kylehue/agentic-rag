import asyncio
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from app.core.config import settings
from app.errors.document import InvalidDocumentError
from app.plugin.base import Plugin
from app.plugin.hooks import (
    HookBus,
    HOOK_FILE_EMITTED,
    HOOK_FILE_SUBPROCESSED,
)
from app.plugin.registry import PluginRegistry
from app.services import ingestion as ingestion_module
from app.services.ingestion import IngestionService
from app.store_file.local import LocalFileStorage
from app.store_sql.local import LocalSqlStorage

from fakes import (
    FakeEmbedder,
    FakeLLM,
    FakeVectorStorage,
    make_table_element,
    make_text_element,
)
from app.plugins.table import TablePlugin
from app.plugins.text import TextPlugin

TABLE_HTML = (
    "<table>"
    "<tr><td>region</td><td>amount</td></tr>"
    "<tr><td>north</td><td>10</td></tr>"
    "</table>"
)

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
    """A plugin that accepts every CSV and saves one marker chunk."""

    @property
    def name(self) -> str:
        return "sidecar"

    def accepts(self, context) -> bool:
        return context.source_filename.endswith(".csv")

    async def process(self, context, runtime):
        from app.models.chunk import IngestedChunk

        chunk = IngestedChunk(plugin=self.name, text="sidecar marker")
        await runtime.save_chunk(chunk)
        return [chunk]


def build_service(tmp_path, *, llm=None, hooks=None, plugins=None):
    llm = llm if llm is not None else FakeLLM()
    sql_storage = LocalSqlStorage(storage_dir=tmp_path / "sql")
    file_storage = LocalFileStorage(storage_dir=tmp_path / "file")
    vector_storage = FakeVectorStorage()
    hooks = hooks if hooks is not None else HookBus()
    registry = PluginRegistry(hooks)
    for plugin in (
        plugins
        if plugins is not None
        else [TextPlugin(), TablePlugin()]
    ):
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
        chunks = await service.ingest_bytes(
            b"region,amount\nnorth,10\n",
            "sales.csv",
            "text/csv",
        )
        documents = await sql_storage.get_all(
            settings.DOCUMENT_METADATA_TABLE_NAME
        )
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

    # The source file was stored and recorded.
    assert len(documents) == 1
    assert documents[0]["file_orig_filename"] == "sales.csv"
    assert Path(documents[0]["file_path"]).is_file()

    # Chunk metadata was persisted by the plugin, without SQL rows.
    assert len(chunk_rows) == 1
    metadata = chunk_rows[0]["metadata"]
    assert metadata["chunk_source_id"] == chunks[0].metadata["chunk_source_id"]
    assert metadata["chunk_table_name"] == "sales"
    assert "sql_rows" not in metadata
    assert Path(metadata["chunk_file_path"]).is_file()
    assert b"north" in Path(metadata["chunk_file_path"]).read_bytes()

    # No per-table data tables were created; only the system tables exist.
    assert db_tables == {
        settings.CHUNK_TABLE_NAME,
        settings.DOCUMENT_METADATA_TABLE_NAME,
    }

    # The vector was added.
    assert vector_storage.added[0][0] == [chunks[0].chunk_id]


def test_text_ingest_emits_table_subprocess(tmp_path):
    llm = FakeLLM(json.dumps(ANALYSIS))
    service, sql_storage, file_storage, vector_storage, hooks = build_service(
        tmp_path,
        llm=llm,
    )

    emitted_events: list[dict] = []
    subprocessed_events: list[dict] = []

    async def on_emitted(**payload):
        emitted_events.append(payload)

    async def on_subprocessed(**payload):
        subprocessed_events.append(payload)

    hooks.register(HOOK_FILE_EMITTED, on_emitted)
    hooks.register(HOOK_FILE_SUBPROCESSED, on_subprocessed)

    elements = [
        make_text_element("Quarterly report body.", page_number=1),
        make_table_element(TABLE_HTML, page_number=1),
    ]

    async def flow():
        await service.initialize()
        with patch.object(
            ingestion_module,
            "partition",
            return_value=elements,
        ):
            chunks = await service.ingest_bytes(
                b"fake document bytes",
                "report.pdf",
                "application/pdf",
            )
        documents = await sql_storage.get_all(
            settings.DOCUMENT_METADATA_TABLE_NAME
        )
        chunk_rows = await sql_storage.get_all(settings.CHUNK_TABLE_NAME)
        await sql_storage.close()
        return chunks, documents, chunk_rows

    chunks, documents, chunk_rows = asyncio.run(flow())

    # Two source documents: the original file and the emitted table file.
    assert len(documents) == 2
    by_filename = {
        document["file_orig_filename"]: document for document in documents
    }
    original = by_filename["report.pdf"]
    emitted = by_filename["report_table_1.csv"]

    # Text chunks plus the subprocessed table chunk.
    text_chunks = [chunk for chunk in chunks if chunk.plugin == "text"]
    table_chunks = [chunk for chunk in chunks if chunk.plugin == "table"]
    assert len(text_chunks) == 1
    assert len(table_chunks) == 1

    # The table chunk references the text document as its parent.
    table_metadata = next(
        row["metadata"] for row in chunk_rows if row["plugin"] == "table"
    )
    assert table_metadata["chunk_parent_source_id"] == original["source_id"]
    assert table_metadata["chunk_source_id"] == emitted["source_id"]
    assert Path(table_metadata["chunk_file_path"]).is_file()

    # Hooks fired for the emission and its subprocess.
    assert len(emitted_events) == 1
    assert emitted_events[0]["emitted_file"].filename == "report_table_1.csv"
    assert emitted_events[0]["context"].source_id == original["source_id"]
    assert len(subprocessed_events) == 1
    assert [chunk.chunk_id for chunk in subprocessed_events[0]["chunks"]] == [
        table_chunks[0].chunk_id
    ]


def test_multiple_accepting_plugins_all_run(tmp_path):
    llm = FakeLLM(json.dumps(ANALYSIS))
    service, sql_storage, *_ = build_service(
        tmp_path,
        llm=llm,
        plugins=[TextPlugin(), TablePlugin(), SidecarPlugin()],
    )

    async def flow():
        await service.initialize()
        chunks = await service.ingest_bytes(
            b"region,amount\nnorth,10\n",
            "sales.csv",
            "text/csv",
        )
        chunk_rows = await sql_storage.get_all(settings.CHUNK_TABLE_NAME)
        await sql_storage.close()
        return chunks, chunk_rows

    chunks, chunk_rows = asyncio.run(flow())

    # TablePlugin and SidecarPlugin both accepted the CSV and both ran.
    table_chunks = [c for c in chunks if c.metadata.get("chunk_table_name")]
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


def test_file_no_plugin_accepts_raises(tmp_path):
    service, sql_storage, *_ = build_service(tmp_path)

    async def flow():
        await service.initialize()
        with pytest.raises(InvalidDocumentError):
            await service.ingest_bytes(b"x", "photo.png", "image/png")
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
            await service.ingest_bytes(
                b"a,b\n1,2",
                "data.csv",
                "text/csv",
            )
        await sql_storage.close()

    asyncio.run(flow())
