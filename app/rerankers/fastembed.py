import asyncio
from collections.abc import Sequence
from dataclasses import replace

from fastembed.rerank.cross_encoder import TextCrossEncoder

from app.models.chunk import RetrievedChunk, RetrievedTextChunk
from app.rerankers.base import Reranker


def _score(
    model: TextCrossEncoder, query: str, texts: list[str], batch_size: int
) -> list[float]:
    """Score each (query, text) pair; fastembed is synchronous (ONNX), so this
    runs in the worker thread."""
    return list(model.rerank(query, texts, batch_size=batch_size))


class FastReranker(Reranker):
    """Local cross-encoder reranking via fastembed (ONNX runtime, no API key).

    The model is loaded eagerly (download + load at construction, into
    `cache_dir`). fastembed is synchronous, so inference runs in a worker
    thread.
    """

    def __init__(
        self,
        *,
        model_name: str,
        batch_size: int = 64,
        cache_dir: str | None = None,
    ) -> None:
        self._batch_size = batch_size
        self._model = TextCrossEncoder(
            model_name, cache_dir=cache_dir, lazy_load=False
        )

    async def rerank(
        self,
        query: str,
        chunks: Sequence[RetrievedChunk],
    ) -> list[RetrievedChunk]:
        if not chunks:
            return []
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Cannot rerank with an empty query")

        # The cross-encoder scores (query, text) pairs, so it can only re-rank
        # text chunks. Image chunks have no text: they keep their retrieval
        # score. (Note the two score scales differ, so re-ranked text chunks
        # generally order above image chunks.)
        text_chunks = [c for c in chunks if isinstance(c, RetrievedTextChunk)]
        image_chunks = [c for c in chunks if not isinstance(c, RetrievedTextChunk)]

        ranked: list[RetrievedChunk] = []
        if text_chunks:
            texts = [chunk.text for chunk in text_chunks]
            scores = await asyncio.to_thread(_score, self._model, query, texts, self._batch_size)
            if len(scores) != len(text_chunks):
                raise RuntimeError(
                    "Reranker returned an unexpected number of scores"
                )
            ranked.extend(
                replace(chunk, score=score)
                for chunk, score in zip(text_chunks, scores)
            )
        ranked.extend(image_chunks)

        ranked.sort(key=lambda chunk: chunk.score, reverse=True)
        return ranked
