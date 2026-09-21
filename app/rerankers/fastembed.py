import asyncio
from collections.abc import Sequence
from dataclasses import replace

from fastembed.rerank.cross_encoder import TextCrossEncoder

from app.models.chunk import RetrievedChunk
from app.rerankers.base import Reranker


def _score(
    model: TextCrossEncoder, query: str, texts: list[str], batch_size: int
) -> list[float]:
    """Score each (query, text) pair; fastembed is synchronous (ONNX), so this
    runs in the worker thread."""
    return list(model.rerank(query, texts, batch_size=batch_size))


class FastReranker(Reranker):
    """Local cross-encoder reranking via fastembed (ONNX runtime, no API key).

    The model is created lazily (fastembed `lazy_load`) so the download and
    load happen on first use, not at construction. fastembed is synchronous,
    so inference runs in a worker thread.
    """

    def __init__(self, *, model_name: str, batch_size: int = 64) -> None:
        self._model_name = model_name
        self._batch_size = batch_size
        self._model: TextCrossEncoder | None = None

    def _ensure_model(self) -> TextCrossEncoder:
        if self._model is None:
            self._model = TextCrossEncoder(self._model_name, lazy_load=True)
        return self._model

    async def rerank(
        self,
        query: str,
        chunks: Sequence[RetrievedChunk],
    ) -> list[RetrievedChunk]:
        if not chunks:
            return []
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Cannot rerank with an empty query")

        model = self._ensure_model()
        texts = [chunk.text for chunk in chunks]
        scores = await asyncio.to_thread(_score, model, query, texts, self._batch_size)
        if len(scores) != len(chunks):
            raise RuntimeError("Reranker returned an unexpected number of scores")

        ranked = sorted(zip(chunks, scores), key=lambda pair: pair[1], reverse=True)
        return [replace(chunk, score=score) for chunk, score in ranked]
