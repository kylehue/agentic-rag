import asyncio
import json
from types import SimpleNamespace

import pytest

from app.database import CHUNK_TABLE_NAME, DOCUMENT_METADATA_TABLE_NAME
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

    async def retrieve(self, user_query, where=None):
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

    asyncio.run(parts.sql_storage.create_tables())

    assert CHUNK_TABLE_NAME in parts.sql_storage.tables
    assert DOCUMENT_METADATA_TABLE_NAME in parts.sql_storage.tables


def test_ingest_end_to_end_through_the_wrapper():
    rag, parts = build_rag()

    async def flow():
        await parts.sql_storage.create_tables()
        return await rag.ingest(
            file_bytes=CSV_BYTES,
            filename="sales.csv",
            content_type="text/csv",
            description="Acme Q3 sales pack.",
            chat_id="chat-1",
        )

    origin_source_id, chunks = asyncio.run(flow())

    # The built-in plugin set (TextPlugin + TablePlugin) handled it.
    assert len(chunks) == 1
    assert chunks[0].plugin == "table"
    assert origin_source_id

    # The file was stored exactly once, with its original bytes and chat.
    assert len(parts.file_storage.files) == 1
    (stored_path, stored_bytes), = parts.file_storage.files.items()
    assert stored_path.startswith("documents/")
    assert stored_bytes == CSV_BYTES

    # The chunk was embedded, vector-added (chat metadata), and recorded
    # (chat column).
    assert parts.vector_storage.added[0][0] == [chunks[0].chunk_id]
    assert parts.vector_storage.added[0][2] == [{"chat_id": "chat-1"}]
    rows = parts.sql_storage.tables[CHUNK_TABLE_NAME]
    assert len(rows) == 1
    assert rows[0]["origin_source_id"] == rows[0]["source_id"] == origin_source_id
    assert rows[0]["chat_id"] == "chat-1"
    documents = parts.sql_storage.tables[DOCUMENT_METADATA_TABLE_NAME]
    assert documents[0]["chat_id"] == "chat-1"

    # The description reached the TablePlugin's LLM.
    assert any(
        "Acme Q3 sales pack." in prompt for prompt in parts.llm.prompts
    )


def test_ingest_without_description_leaves_prompt_unchanged():
    rag, parts = build_rag()

    async def flow():
        await parts.sql_storage.create_tables()
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
        await parts.sql_storage.create_tables()
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


class ScopedRetriever(Retriever):
    """Stands in for the real retrievers: applies the where condition at
    the index (what Vector/SparseRetriever do with the where map)."""

    def __init__(self, chunks: list[RetrievedChunk]) -> None:
        self._chunks = chunks
        self.wheres: list = []

    async def retrieve(self, user_query, where=None):
        self.wheres.append(where)
        if where is None:
            return self._chunks
        return [
            chunk
            for chunk in self._chunks
            if all(getattr(chunk, column) == value for column, value in where.items())
        ]


def test_retrieve_scoped_to_a_chat():
    chunks = [
        RetrievedChunk(
            chunk_id="c1",
            source_id="s1",
            origin_source_id="s1",
            plugin="text",
            text="one",
            score=0.9,
            chat_id="chat-a",
        ),
        RetrievedChunk(
            chunk_id="c2",
            source_id="s2",
            origin_source_id="s2",
            plugin="text",
            text="two",
            score=0.8,
            chat_id="chat-b",
        ),
    ]
    retriever = ScopedRetriever(chunks)
    rag, _ = build_rag(retriever=retriever)

    # A chat scope reaches the retriever and keeps only that chat's chunks...
    scoped = asyncio.run(rag.retrieve("q", chat_id="chat-a"))
    assert [chunk.chunk_id for chunk in scoped] == ["c1"]
    # The chat id reached the retriever as the where condition map.
    assert retriever.wheres == [{"chat_id": "chat-a"}]
    # ...and no chat is unbounded.
    assert [chunk.chunk_id for chunk in asyncio.run(rag.retrieve("q"))] == [
        "c1",
        "c2",
    ]
    assert retriever.wheres[-1] is None


