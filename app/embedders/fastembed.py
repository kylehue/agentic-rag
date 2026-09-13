import asyncio
from collections.abc import Sequence

from fastembed import TextEmbedding

from app.core.config import Settings
from app.embedders.base import Embedder


def _collect_vectors(vectors) -> list:
    """Consume a fastembed generator into a list; the ONNX inference happens
    while consuming, so this runs in the worker thread."""
    return list(vectors)


class FastEmbedder(Embedder):
    """Local embeddings via fastembed (ONNX runtime, no API key).

    The model is created lazily with fastembed's `lazy_load`, so the model
    download and load happen on first use, not at construction. fastembed is
    synchronous, so inference runs in a worker thread.
    """

    def __init__(self, settings: Settings | None = None):
        settings = settings or Settings()
        self._model_name = settings.FASTEMBED_MODEL
        self._batch_size = settings.EMBEDDING_BATCH_SIZE
        self._model: TextEmbedding | None = None

    def _ensure_model(self) -> TextEmbedding:
        if self._model is None:
            self._model = TextEmbedding(self._model_name, lazy_load=True)
        return self._model

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        if any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError("Cannot embed empty text")

        model = self._ensure_model()
        vectors = await asyncio.to_thread(
            _collect_vectors, model.embed(list(texts), self._batch_size)
        )
        embeddings = [vector.tolist() for vector in vectors]
        if len(embeddings) != len(texts):
            raise RuntimeError(
                "Embedding provider returned an unexpected number of vectors"
            )
        return embeddings

    async def embed_query(self, text: str) -> list[float]:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Cannot embed empty text")

        model = self._ensure_model()
        # query_embed applies the model's query instruction, which asymmetric
        # models (such as the bge family) expect on queries.
        vectors = await asyncio.to_thread(_collect_vectors, model.query_embed(text))
        if len(vectors) != 1:
            raise RuntimeError(
                "Embedding provider returned an unexpected number of vectors"
            )
        return vectors[0].tolist()
