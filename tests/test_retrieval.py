import asyncio
from dataclasses import replace

import pytest
from pydantic import ValidationError

from app.api_schemas.chunk import RetrievedChunkSchema
from app.models.chunk import RetrievedChunk, RetrievedTextChunk
from app.plugin.base import Plugin
from app.plugin.registry import PluginRegistry
from app.rerankers.base import Reranker
from app.retrievers.base import Retriever
from app.services.retrieval import RetrievalService

from fakes import FakeLLM


class FakeRetriever(Retriever):
    def __init__(self, chunks: list[RetrievedChunk]) -> None:
        self._chunks = chunks

    async def retrieve(self, user_query, where=None):
        return self._chunks


class EnrichingPlugin(Plugin):
    """Finalizes only the chunks it produced."""

    def __init__(self, name: str) -> None:
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def accepts(self, context) -> bool:
        return False

    async def on_retrieval_finalize(self, chunk, context, runtime):
        if chunk.plugin != self._name:
            return None
        return replace(
            chunk, text=f"{chunk.text} [enriched with {context.user_query}]"
        )


def make_chunk(plugin: str, chunk_id: str = "c1") -> RetrievedChunk:
    return RetrievedTextChunk(
        chunk_id=chunk_id,
        source_id="s1",
        origin_source_id="s1",
        plugin=plugin,
        text=f"text from {plugin}",
        score=0.5,
    )


def build_service(registry, retriever, reranker=None, top_k=5):
    return RetrievalService(
        retriever=retriever,
        llm=FakeLLM(),
        registry=registry,
        reranker=reranker,
        top_k=top_k,
    )


def test_finalize_hook_enriches_only_owning_chunks():
    registry = PluginRegistry()
    registry.register(EnrichingPlugin("enricher"))

    service = build_service(
        registry,
        FakeRetriever([make_chunk("enricher", "c1"), make_chunk("text", "c2")]),
    )

    results = asyncio.run(service.retrieve("q"))

    assert results[0].text == "text from enricher [enriched with q]"
    assert results[1].text == "text from text"


def test_chunk_without_finalizing_plugin_passes_through():
    service = build_service(
        PluginRegistry(),
        FakeRetriever([make_chunk("unknown")]),
    )

    results = asyncio.run(service.retrieve("q"))

    assert results[0].text == "text from unknown"


class FakeReranker(Reranker):
    """Reverses the candidate order and stamps a marker score, so a test can
    prove the service ran the reranker (and on what it was given)."""

    def __init__(self, score: float = 0.99) -> None:
        self._score = score
        self.calls: list[tuple[str, list[RetrievedChunk]]] = []

    async def rerank(self, query, chunks):
        self.calls.append((query, list(chunks)))
        return [replace(chunk, score=self._score) for chunk in list(chunks)[::-1]]


def test_retrieval_service_applies_the_reranker():
    chunks = [make_chunk("text", "c1"), make_chunk("text", "c2"), make_chunk("text", "c3")]
    reranker = FakeReranker()
    service = build_service(PluginRegistry(), FakeRetriever(chunks), reranker)

    results = asyncio.run(service.retrieve("q"))

    # The reranker reversed the order and stamped its marker score...
    assert [chunk.chunk_id for chunk in results] == ["c3", "c2", "c1"]
    assert all(chunk.score == 0.99 for chunk in results)
    # ...and it ran on the retriever's candidates, in the retriever's order.
    assert reranker.calls == [("q", chunks)]


def test_retrieval_without_a_reranker_keeps_the_retriever_order():
    chunks = [make_chunk("text", "c1"), make_chunk("text", "c2")]
    service = build_service(PluginRegistry(), FakeRetriever(chunks))

    results = asyncio.run(service.retrieve("q"))

    assert [chunk.chunk_id for chunk in results] == ["c1", "c2"]
    assert all(chunk.score == 0.5 for chunk in results)


def test_retrieval_service_trims_the_candidate_pool_to_top_k():
    chunks = [make_chunk("text", f"c{i}") for i in range(7)]
    service = build_service(PluginRegistry(), FakeRetriever(chunks), top_k=3)

    results = asyncio.run(service.retrieve("q"))

    assert len(results) == 3
    assert [chunk.chunk_id for chunk in results] == ["c0", "c1", "c2"]


def test_reranker_runs_before_the_top_k_trim():
    chunks = [make_chunk("text", f"c{i}") for i in range(7)]
    # FakeReranker reverses the pool, so the final top_k is the pool's tail.
    service = build_service(
        PluginRegistry(), FakeRetriever(chunks), FakeReranker(), top_k=3
    )

    results = asyncio.run(service.retrieve("q"))

    assert [chunk.chunk_id for chunk in results] == ["c6", "c5", "c4"]
    assert all(chunk.score == 0.99 for chunk in results)


def test_retrieval_dedups_chunks_that_share_a_key():
    # Two chunks share key "img" (an image and its description); one has a
    # null key. The retriever returns them best-first.
    chunks = [
        RetrievedTextChunk(
            chunk_id="img-desc", source_id="s", origin_source_id="s",
            plugin="text", text="the description", score=0.9, key="img",
        ),
        RetrievedTextChunk(
            chunk_id="other", source_id="s2", origin_source_id="s2",
            plugin="text", text="other", score=0.7, key=None,
        ),
        RetrievedTextChunk(
            chunk_id="img-img", source_id="s", origin_source_id="s",
            plugin="image", text="", score=0.5, key="img",
        ),
    ]
    service = build_service(PluginRegistry(), FakeRetriever(chunks), top_k=5)

    results = asyncio.run(service.retrieve("q"))

    # The better "img" chunk survives, its duplicate is dropped, and the
    # null-key chunk is kept.
    assert [chunk.chunk_id for chunk in results] == ["img-desc", "other"]


def test_retrieval_dedup_still_respects_top_k():
    chunks = [
        RetrievedTextChunk(
            chunk_id=f"c{i}", source_id="s", origin_source_id="s",
            plugin="text", text="t", score=1.0 - i * 0.1, key=f"k{i}",
        )
        for i in range(4)
    ]
    service = build_service(PluginRegistry(), FakeRetriever(chunks), top_k=2)

    results = asyncio.run(service.retrieve("q"))

    assert [chunk.chunk_id for chunk in results] == ["c0", "c1"]


def test_from_dict_requires_origin_source_id():
    data = {
        "chunk_id": "c1",
        "source_id": "s1",
        "origin_source_id": "s1",
        "plugin": "text",
        "text": "t",
        "parent_source_id": None,
    }

    chunk = RetrievedChunk.from_dict(data, score=0.4)
    assert chunk.origin_source_id == "s1"
    assert chunk.parent_source_id is None

    without_origin = {k: v for k, v in data.items() if k != "origin_source_id"}
    with pytest.raises(KeyError):
        RetrievedChunk.from_dict(without_origin)


def test_retrieved_chunk_schema_rejects_missing_origin():
    chunk = make_chunk("text")

    # A well-formed chunk (origin always set) validates.
    schema = RetrievedChunkSchema.model_validate(chunk)
    assert schema.origin_source_id == "s1"

    # A chunk without an origin is invalid.
    broken = replace(chunk, origin_source_id=None)  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        RetrievedChunkSchema.model_validate(broken)


def test_retrieval_completed_fires_with_finalized_chunks():
    registry = PluginRegistry()
    events: list = []

    class CompletionObserver(Plugin):
        @property
        def name(self) -> str:
            return "completion-observer"

        def accepts(self, context) -> bool:
            return False

        async def on_retrieval_completed(self, chunks, context, runtime) -> None:
            events.append((context, chunks, runtime))

    registry.register(CompletionObserver())

    service = build_service(registry, FakeRetriever([make_chunk("text")]))

    asyncio.run(service.retrieve("q"))

    assert len(events) == 1
    context, chunks, runtime = events[0]
    assert len(chunks) == 1
    # The event carries the per-run retrieval context and runtime.
    assert context.user_query == "q"
    assert runtime.context is context


class RecordingRetriever(Retriever):
    def __init__(self):
        self.wheres: list = []

    async def retrieve(self, user_query, where=None):
        self.wheres.append(where)
        return []


def test_retrieve_passes_the_where_condition_to_the_retriever():
    registry = PluginRegistry()
    retriever = RecordingRetriever()
    service = build_service(registry, retriever)

    asyncio.run(service.retrieve("q", where={"chat_id": "c1"}))
    asyncio.run(service.retrieve("q"))

    assert retriever.wheres == [{"chat_id": "c1"}, None]


def test_sparse_retriever_applies_the_where_condition_in_the_real_fts_search(
    tmp_path,
):
    """Regression: the FTS search used to alias the table while the
    condition was compiled against the real name, so any scoped lexical
    search failed with 'no such column'."""
    from app.database import CHUNK_TABLE_NAME
    from app.retrievers.sparse import SparseRetriever
    from app.store_sql.local import LocalSqlStorage

    async def flow():
        storage = LocalSqlStorage(storage_dir=tmp_path)
        await storage.create_tables()
        await storage.upsert(
            CHUNK_TABLE_NAME,
            [
                {
                    "chunk_id": "c1",
                    "source_id": "s1",
                    "origin_source_id": "s1",
                    "plugin": "text",
                    "text": "the tomatoes weighed forty pounds",
                    "chat_id": "chat-a",
                },
                {
                    "chunk_id": "c2",
                    "source_id": "s2",
                    "origin_source_id": "s2",
                    "plugin": "text",
                    "text": "the tomatoes weighed thirty pounds",
                    "chat_id": "chat-b",
                },
            ],
        )
        try:
            retriever = SparseRetriever(sql_storage=storage, top_k=5)
            scoped = await retriever.retrieve("weighed", where={"chat_id": "chat-a"})
            unscoped = await retriever.retrieve("weighed")
            return scoped, unscoped
        finally:
            await storage.close()

    scoped, unscoped = asyncio.run(flow())

    assert [chunk.chunk_id for chunk in scoped] == ["c1"]
    assert {chunk.chat_id for chunk in scoped} == {"chat-a"}
    assert {chunk.chunk_id for chunk in unscoped} == {"c1", "c2"}
