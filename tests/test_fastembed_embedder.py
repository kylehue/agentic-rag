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

    def embed(self, documents, batch_size=256, **kwargs):
        self.document_batches.append(list(documents))
        for document in documents:
            yield np.array([float(len(document))], dtype=np.float32)


def make_embedder(monkeypatch) -> tuple[FastEmbedder, StubModel]:
    stub = StubModel()
    monkeypatch.setattr(
        fastembed_module, "TextEmbedding", lambda *args, **kwargs: stub
    )
    return FastEmbedder(model_name="stub", batch_size=1), stub


def test_embed_text_returns_one_vector_per_text_in_order(monkeypatch):
    embedder, stub = make_embedder(monkeypatch)

    out = asyncio.run(embedder.embed_text(["aa", "bbb", "cccc"]))

    assert out == [[2.0], [3.0], [4.0]]
    # All texts went out in one batch, in order.
    assert stub.document_batches == [["aa", "bbb", "cccc"]]


def test_embed_text_rejects_empty_text(monkeypatch):
    embedder, _ = make_embedder(monkeypatch)

    with pytest.raises(ValueError):
        asyncio.run(embedder.embed_text(["ok", "   "]))


def test_embed_text_empty_input_is_empty_output(monkeypatch):
    embedder, _ = make_embedder(monkeypatch)

    assert asyncio.run(embedder.embed_text([])) == []


def test_embed_text_is_used_for_queries_too(monkeypatch):
    # With no separate query method, a single-text embed_text call is how a
    # query is embedded.
    embedder, stub = make_embedder(monkeypatch)

    out = asyncio.run(embedder.embed_text(["what?"]))

    assert out == [[5.0]]
    assert stub.document_batches == [["what?"]]


def test_model_is_created_eagerly(monkeypatch):
    created: list = []

    def factory(*args, **kwargs):
        created.append((args, kwargs))
        return StubModel()

    monkeypatch.setattr(fastembed_module, "TextEmbedding", factory)
    embedder = FastEmbedder(
        model_name="stub", batch_size=1, cache_dir="/tmp/models"
    )
    # The model loads at construction (not on first use), into the cache dir.
    assert len(created) == 1
    assert created[0][0] == ("stub",)
    assert created[0][1]["cache_dir"] == "/tmp/models"
    assert created[0][1]["lazy_load"] is False
    asyncio.run(embedder.embed_text("x"))
    assert len(created) == 1  # no second load on use
