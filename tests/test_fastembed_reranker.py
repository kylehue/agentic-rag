import asyncio

import pytest

from app.models.chunk import RetrievedChunk
from app.rerankers import fastembed as fastembed_rerank_module
from app.rerankers.fastembed import FastReranker


def make_chunk(text: str, chunk_id: str) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        source_id="s1",
        origin_source_id="s1",
        plugin="text",
        text=text,
        score=0.5,
    )


class StubCrossEncoder:
    """A stand-in for fastembed's TextCrossEncoder: records what it was called
    with and returns fixed per-document scores."""

    instances: list["StubCrossEncoder"] = []

    def __init__(self, model_name, lazy_load=False, **kwargs):
        self.model_name = model_name
        self.lazy_load = lazy_load
        self.rerank_calls: list[tuple[str, list, int]] = []
        StubCrossEncoder.instances.append(self)

    def rerank(self, query, documents, batch_size=64, **kwargs):
        self.rerank_calls.append((query, list(documents), batch_size))
        # Positional scores so the reordering is deterministic and checkable:
        # for three docs -> [1.0, 3.0, 2.0] (middle doc most relevant).
        return [1.0, 3.0, 2.0][: len(documents)]


def make_reranker(monkeypatch) -> tuple[FastReranker, StubCrossEncoder]:
    StubCrossEncoder.instances = []
    monkeypatch.setattr(
        fastembed_rerank_module,
        "TextCrossEncoder",
        lambda *args, **kwargs: StubCrossEncoder(*args, **kwargs),
    )
    reranker = FastReranker(model_name="stub-model", batch_size=8)
    # Force the lazy model to load (an empty rerank returns before loading it).
    asyncio.run(reranker.rerank("load", [make_chunk("x", "x")]))
    stub = StubCrossEncoder.instances[0]
    stub.rerank_calls.clear()  # drop the loading call; tests assert their own
    return reranker, stub


def test_rerank_reorders_and_rescores_chunks(monkeypatch):
    reranker, stub = make_reranker(monkeypatch)
    chunks = [
        make_chunk("doc1", "c1"),
        make_chunk("doc2", "c2"),
        make_chunk("doc3", "c3"),
    ]

    results = asyncio.run(reranker.rerank("q", chunks))

    # Reordered by descending score (c2=3.0, c3=2.0, c1=1.0); scores replaced.
    assert [chunk.chunk_id for chunk in results] == ["c2", "c3", "c1"]
    assert [chunk.score for chunk in results] == [3.0, 2.0, 1.0]
    # The same set of chunks, otherwise untouched.
    assert {chunk.chunk_id for chunk in results} == {"c1", "c2", "c3"}
    assert stub.rerank_calls == [("q", ["doc1", "doc2", "doc3"], 8)]


def test_rerank_empty_is_empty(monkeypatch):
    reranker, _ = make_reranker(monkeypatch)

    assert asyncio.run(reranker.rerank("q", [])) == []


def test_rerank_rejects_an_empty_query(monkeypatch):
    reranker, _ = make_reranker(monkeypatch)

    with pytest.raises(ValueError):
        asyncio.run(reranker.rerank("   ", [make_chunk("doc", "c1")]))


def test_model_is_created_lazily_and_configured(monkeypatch):
    StubCrossEncoder.instances = []
    monkeypatch.setattr(
        fastembed_rerank_module,
        "TextCrossEncoder",
        lambda *args, **kwargs: StubCrossEncoder(*args, **kwargs),
    )
    FastReranker(model_name="stub-model", batch_size=8)
    assert StubCrossEncoder.instances == []  # construction does not load the model

    reranker = FastReranker(model_name="stub-model", batch_size=8)
    asyncio.run(reranker.rerank("q", [make_chunk("doc", "c1")]))
    assert len(StubCrossEncoder.instances) == 1
    assert StubCrossEncoder.instances[0].model_name == "stub-model"
    assert StubCrossEncoder.instances[0].lazy_load is True
