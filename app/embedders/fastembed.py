import asyncio
from collections.abc import Sequence

from fastembed import TextEmbedding

from app.embedders.base import TextEmbedder


def _collect_vectors(vectors) -> list:
    """Consume a fastembed generator into a list; the ONNX inference happens
    while consuming, so this runs in the worker thread."""
    return list(vectors)


class FastEmbedder(TextEmbedder):
    """Local text embeddings via fastembed (ONNX runtime, no API key).

    The model is loaded eagerly (download + load at construction, into
    `cache_dir`). fastembed is synchronous, so inference runs in a worker
    thread.
    """

    def __init__(
        self,
        *,
        model_name: str,
        batch_size: int,
        cache_dir: str | None = None,
    ):
        self._batch_size = batch_size
        self._model = TextEmbedding(
            model_name, cache_dir=cache_dir, lazy_load=False
        )

    async def embed_text(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        if any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError("Cannot embed empty text")

        vectors = await asyncio.to_thread(
            _collect_vectors, self._model.embed(list(texts), self._batch_size)
        )
        embeddings = [vector.tolist() for vector in vectors]
        if len(embeddings) != len(texts):
            raise RuntimeError(
                "Embedding provider returned an unexpected number of vectors"
            )
        return embeddings
