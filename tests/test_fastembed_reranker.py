import asyncio

import pytest

from app.models.chunk import RetrievedChunk, RetrievedImageChunk, RetrievedTextChunk
from app.rerankers import fastembed as fastembed_rerank_module
from app.rerankers.fastembed import FastReranker


def make_chunk(text: str, chunk_id: str) -> RetrievedChunk:
    return RetrievedTextChunk(
        chunk_id=chunk_id,
        source_id="s1",
        origin_source_id="s1",
        plugin="text",
        text=text,
        score=0.5,
    )


def make_image_chunk(chunk_id: str) -> RetrievedChunk:
    return RetrievedImageChunk(
        chunk_id=chunk_id,
        source_id="s1",
        origin_source_id="s1",
        plugin="image",
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
    # The model loads eagerly at construction (the stub is created then).
    reranker = FastReranker(model_name="stub-model", batch_size=8)
    stub = StubCrossEncoder.instances[0]
    return reranker, stub


def test_rerank_reorders_and_rescores_chunks(monkeypatch):
    reranker, stub = make_reranker(monkeypatch)
    chunks = [
        make_chunk("doc1", "c1"),
        make_chunk("doc2", "c2"),
        make_chunk("doc3", "c3"),
    ]

    results = asyncio.run(reranker.rerank("q", chunks))

    # Reordered by the cross-encoder ranking (c2 > c3 > c1); the stored score
    # is the RRF score of that rank (1/(60+rank)), not the raw score.
    assert [chunk.chunk_id for chunk in results] == ["c2", "c3", "c1"]
    assert [chunk.score for chunk in results] == [1 / 61, 1 / 62, 1 / 63]
    # The same set of chunks, otherwise untouched.
    assert {chunk.chunk_id for chunk in results} == {"c1", "c2", "c3"}
    assert stub.rerank_calls == [("q", ["doc1", "doc2", "doc3"], 8)]


def test_rerank_fuses_text_and_image_rankings_with_rrf(monkeypatch):
    reranker, _ = make_reranker(monkeypatch)
    # c1 scores 1.0, c2 scores 3.0 (positional stub), so the cross-encoder
    # re-ranks the text as c2 > c1. The images keep the retrievers' order
    # (i1 > i2). RRF interleaves the two rankings by rank.
    chunks = [make_chunk("doc1", "c1"), make_chunk("doc2", "c2"), make_image_chunk("i1"), make_image_chunk("i2")]

    results = asyncio.run(reranker.rerank("q", chunks))

    assert [chunk.chunk_id for chunk in results] == ["c2", "i1", "c1", "i2"]
    # Only the text chunks are scored by the cross-encoder; the images pass
    # through untouched (no image text is ever sent to the model).
    # (scores are RRF: c2/i1 tie at 1/61, c1/i2 tie at 1/62, text first)
    assert [round(chunk.score, 6) for chunk in results] == [
        round(1 / 61, 6),
        round(1 / 61, 6),
        round(1 / 62, 6),
        round(1 / 62, 6),
    ]


def test_rerank_empty_is_empty(monkeypatch):
    reranker, _ = make_reranker(monkeypatch)

    assert asyncio.run(reranker.rerank("q", [])) == []


def test_rerank_rejects_an_empty_query(monkeypatch):
    reranker, _ = make_reranker(monkeypatch)

    with pytest.raises(ValueError):
        asyncio.run(reranker.rerank("   ", [make_chunk("doc", "c1")]))


def test_model_is_created_eagerly_and_configured(monkeypatch):
    StubCrossEncoder.instances = []
    monkeypatch.setattr(
        fastembed_rerank_module,
        "TextCrossEncoder",
        lambda *args, **kwargs: StubCrossEncoder(*args, **kwargs),
    )
    FastReranker(model_name="stub-model", batch_size=8, cache_dir="/tmp/models")
    # The model loads at construction, not on first use, into the cache dir.
    assert len(StubCrossEncoder.instances) == 1
    assert StubCrossEncoder.instances[0].model_name == "stub-model"
    assert StubCrossEncoder.instances[0].lazy_load is False
