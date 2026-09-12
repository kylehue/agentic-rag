import asyncio
import json
from types import SimpleNamespace

import pytest

from app.core.config import settings
from app.errors.document import InvalidDocumentError
from app.models.chunk import RetrievedChunk
from app.plugins.table import TablePlugin
from app.plugins.text import TextPlugin
from app.retrievers.base import Retriever
from app.services.rag import RagService

from fakes import (
    FakeEmbedder,
    FakeFileStorage,
    FakeLLM,
    FakeSqlStorage,
    FakeVectorStorage,
)

ANALYSIS = {
    "workbook_description": "Regional sales figures.",
    "tables": [
        {
            "index": 0,
            "description": "Sales per region.",
        }
    ],
}

CSV_BYTES = b"region,amount\nnorth,10\n"


class NoopRetriever(Retriever):
    def __init__(self, chunks: list[RetrievedChunk] | None = None) -> None:
        self._chunks = chunks if chunks is not None else []

    async def retrieve(self, user_query):
        return self._chunks


def build_rag(llm=None, retriever=None, plugins=None):
    """Build a RagService on fakes; returns (rag, parts) for assertions.

    The RagService registers no plugins by default, so the built-ins are
    passed explicitly here (as the container does).
    """
    llm = llm if llm is not None else FakeLLM(json.dumps(ANALYSIS))
    parts = SimpleNamespace(
        llm=llm,
        retriever=retriever if retriever is not None else NoopRetriever(),
        embedder=FakeEmbedder(),
        vector_storage=FakeVectorStorage(),
        sql_storage=FakeSqlStorage(),
        file_storage=FakeFileStorage(),
    )
    rag = RagService(
        llm=parts.llm,
        embedder=parts.embedder,
        retriever=parts.retriever,
        vector_storage=parts.vector_storage,
        sql_storage=parts.sql_storage,
        file_storage=parts.file_storage,
        plugins=plugins
        if plugins is not None
        else [TextPlugin(), TablePlugin()],
    )
    return rag, parts


def test_initialize_creates_the_system_tables():
    rag, parts = build_rag()

    asyncio.run(rag.initialize())

    assert settings.CHUNK_TABLE_NAME in parts.sql_storage.tables
    assert settings.DOCUMENT_METADATA_TABLE_NAME in parts.sql_storage.tables


def test_ingest_end_to_end_through_the_wrapper():
    rag, parts = build_rag()

    async def flow():
        await rag.initialize()
        return await rag.ingest(
            file_bytes=CSV_BYTES,
            filename="sales.csv",
            content_type="text/csv",
            description="Acme Q3 sales pack.",
        )

    chunks = asyncio.run(flow())

    # The built-in plugin set (TextPlugin + TablePlugin) handled it.
    assert len(chunks) == 1
    assert chunks[0].plugin == "table"

    # The file was stored exactly once, with its original bytes.
    assert len(parts.file_storage.files) == 1
    (stored_path, stored_bytes), = parts.file_storage.files.items()
    assert stored_path.startswith("documents/")
    assert stored_bytes == CSV_BYTES

    # The chunk was embedded, vector-added, and recorded.
    assert parts.vector_storage.added[0][0] == [chunks[0].chunk_id]
    rows = parts.sql_storage.tables[settings.CHUNK_TABLE_NAME]
    assert len(rows) == 1
    assert rows[0]["origin_source_id"] == rows[0]["source_id"]

    # The description reached the TablePlugin's LLM.
    assert any(
        "Acme Q3 sales pack." in prompt for prompt in parts.llm.prompts
    )


def test_ingest_without_description_leaves_prompt_unchanged():
    rag, parts = build_rag()

    async def flow():
        await rag.initialize()
        await rag.ingest(
            file_bytes=CSV_BYTES,
            filename="sales.csv",
            content_type="text/csv",
        )

    asyncio.run(flow())

    assert all(
        "Additional context about the source document" not in prompt
        for prompt in parts.llm.prompts
    )


def test_plugins_option_limits_registered_plugins():
    rag, parts = build_rag(plugins=[TextPlugin()])

    async def flow():
        await rag.initialize()
        await rag.ingest(
            file_bytes=CSV_BYTES,
            filename="sales.csv",
            content_type="text/csv",
        )

    # Only the TextPlugin is registered: the CSV is accepted by nobody.
    with pytest.raises(InvalidDocumentError):
        asyncio.run(flow())


def test_retrieve_delegates_through_the_wrapper():
    chunk = RetrievedChunk(
        chunk_id="c1",
        source_id="s1",
        origin_source_id="s1",
        plugin="table",
        text="evidence",
        score=0.9,
    )
    rag, _ = build_rag(retriever=NoopRetriever([chunk]))

    results = asyncio.run(rag.retrieve("anything"))

    assert results == [chunk]


