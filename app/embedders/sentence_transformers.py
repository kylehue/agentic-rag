import asyncio
from collections.abc import Sequence

from sentence_transformers import SentenceTransformer

from app.embedders.base import Embedder


class SentenceTransformerEmbedder(Embedder):
    def __init__(self, *, model_name: str, hf_token: str, batch_size: int):
        self._model_name = model_name
        self._hf_token = hf_token
        self._batch_size = batch_size
        self._model: SentenceTransformer | None = None
        self._lock = asyncio.Lock()

    def _ensure_model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(
                self._model_name,
                token=self._hf_token or None,
            )
        return self._model

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return await self._embed(list(texts))

    async def embed_query(self, text: str) -> list[float]:
        return (await self._embed([text]))[0]

    async def _embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError("Cannot embed empty text")
        if self._batch_size < 1:
            raise ValueError("EMBEDDING_BATCH_SIZE must be at least 1")

        async with self._lock:
            model = await asyncio.to_thread(self._ensure_model)
            embeddings = await asyncio.to_thread(
                model.encode,
                texts,
                batch_size=self._batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
            )

        vectors = [vector.tolist() for vector in embeddings]

        if len(vectors) != len(texts):
            raise RuntimeError(
                "Embedding provider returned an unexpected number of vectors"
            )
        return vectors
