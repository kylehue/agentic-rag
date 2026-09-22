import asyncio
from collections.abc import Sequence
from dataclasses import replace

from fastembed.rerank.cross_encoder import TextCrossEncoder

from app.models.chunk import RetrievedChunk, RetrievedTextChunk
from app.rerankers.base import Reranker
from app.utils.ranking import rrf


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
        # text chunks. Image chunks have no text to score, so they keep the
        # ranking the retrievers produced. The two are on different scales
        # (cross-encoder scores vs. the retrievers' RRF scores), so they are
        # combined by RRF over their ranks, never by mixing raw scores.
        text_chunks = [c for c in chunks if isinstance(c, RetrievedTextChunk)]
        image_chunks = [c for c in chunks if not isinstance(c, RetrievedTextChunk)]

        text_ranking: list[RetrievedChunk] = []
        if text_chunks:
            texts = [chunk.text for chunk in text_chunks]
            scores = await asyncio.to_thread(_score, self._model, query, texts, self._batch_size)
            if len(scores) != len(text_chunks):
                raise RuntimeError(
                    "Reranker returned an unexpected number of scores"
                )
            # The cross-encoder ranking, best to worst.
            text_ranking = [
                chunk
                for chunk, _ in sorted(
                    zip(text_chunks, scores), key=lambda pair: pair[1], reverse=True
                )
            ]

        # image_chunks is already best-to-worst (the retrievers' order), so the
        # two rankings fuse cleanly on one (RRF) scale.
        fused = rrf([text_ranking, image_chunks], id_fn=lambda c: c.chunk_id)
        return [replace(chunk, score=score) for chunk, score in fused]
