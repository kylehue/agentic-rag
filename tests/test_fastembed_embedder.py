import asyncio

import numpy as np
import pytest

from app.embedders import fastembed as fastembed_module
from app.embedders.fastembed import FastEmbedder


class StubModel:
    """A stand-in for fastembed.TextEmbedding: yields one vector per input
    and records what it was called with."""

    def __init__(self):
        self.document_batches: list = []
        self.queries: list = []

    def embed(self, documents, batch_size=256, **kwargs):
        self.document_batches.append(list(documents))
        for document in documents:
            yield np.array([float(len(document))], dtype=np.float32)

    def query_embed(self, query, **kwargs):
        self.queries.append(query)
        yield np.array([9.0], dtype=np.float32)


def make_embedder(monkeypatch) -> tuple[FastEmbedder, StubModel]:
    stub = StubModel()
    monkeypatch.setattr(
        fastembed_module, "TextEmbedding", lambda *args, **kwargs: stub
    )
    return FastEmbedder(), stub


def test_embed_documents_returns_one_vector_per_text_in_order(monkeypatch):
    embedder, stub = make_embedder(monkeypatch)

    out = asyncio.run(embedder.embed_documents(["aa", "bbb", "cccc"]))

    assert out == [[2.0], [3.0], [4.0]]
    # All texts went out in one batch, in order.
    assert stub.document_batches == [["aa", "bbb", "cccc"]]


def test_embed_documents_rejects_empty_text(monkeypatch):
    embedder, _ = make_embedder(monkeypatch)

    with pytest.raises(ValueError):
        asyncio.run(embedder.embed_documents(["ok", "   "]))


def test_embed_documents_empty_input_is_empty_output(monkeypatch):
    embedder, _ = make_embedder(monkeypatch)

    assert asyncio.run(embedder.embed_documents([])) == []


def test_embed_query_uses_query_embed(monkeypatch):
    embedder, stub = make_embedder(monkeypatch)

    out = asyncio.run(embedder.embed_query("what?"))

    assert out == [9.0]
    assert stub.queries == ["what?"]


def test_model_is_created_lazily(monkeypatch):
    created: list = []

    def factory(*args, **kwargs):
        created.append(args)
        return StubModel()

    monkeypatch.setattr(fastembed_module, "TextEmbedding", factory)
    FastEmbedder()
    assert created == []  # construction does not load the model

    asyncio.run(FastEmbedder().embed_query("x"))
    assert len(created) == 1
