import asyncio
import io

import numpy as np
from PIL import Image

from app.embedders import clip as clip_module
from app.embedders.clip import ClipImageEmbedder
from app.models.content import ImageContent


def _png_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (1, 1), color="red").save(buffer, format="PNG")
    return buffer.getvalue()


class StubTextModel:
    def __init__(self):
        self.batches: list = []

    def embed(self, texts, batch_size=32, **kwargs):
        self.batches.append(list(texts))
        for _ in texts:
            yield np.array([1.0, 1.0])


class StubImageModel:
    def __init__(self):
        self.batches: list = []

    def embed(self, images, batch_size=32, **kwargs):
        self.batches.append(list(images))
        for _ in images:
            yield np.array([2.0, 2.0])


def make_embedder(monkeypatch):
    text_stub, image_stub = StubTextModel(), StubImageModel()
    monkeypatch.setattr(clip_module, "TextEmbedding", lambda *a, **k: text_stub)
    monkeypatch.setattr(clip_module, "ImageEmbedding", lambda *a, **k: image_stub)
    return ClipImageEmbedder(text_model="t", image_model="i"), text_stub, image_stub


def test_embed_text_uses_the_text_model(monkeypatch):
    embedder, text_stub, image_stub = make_embedder(monkeypatch)

    out = asyncio.run(embedder.embed_text(["a", "bb"]))

    assert out == [[1.0, 1.0], [1.0, 1.0]]
    assert text_stub.batches == [["a", "bb"]]
    assert image_stub.batches == []


def test_embed_images_uses_the_image_model(monkeypatch):
    embedder, text_stub, image_stub = make_embedder(monkeypatch)
    images = [ImageContent(data=_png_bytes(), mime_type="image/png")]

    out = asyncio.run(embedder.embed_images(images))

    assert out == [[2.0, 2.0]]
    assert len(image_stub.batches[0]) == 1
    assert text_stub.batches == []


def test_models_are_created_eagerly(monkeypatch):
    created: list = []
    monkeypatch.setattr(
        clip_module,
        "TextEmbedding",
        lambda *a, **k: (created.append("text"), StubTextModel())[1],
    )
    monkeypatch.setattr(
        clip_module,
        "ImageEmbedding",
        lambda *a, **k: (created.append("image"), StubImageModel())[1],
    )
    embedder = ClipImageEmbedder(
        text_model="t", image_model="i", cache_dir="/tmp/models"
    )
    # Both models load at construction (not on first use).
    assert created == ["text", "image"]

    asyncio.run(embedder.embed_text(["a"]))
    assert created == ["text", "image"]  # no further loads on use
