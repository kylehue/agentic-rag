import asyncio
from collections.abc import Sequence

from google import genai
from google.genai import types

from app.core.config import Settings
from app.embedders.base import Embedder


class GeminiEmbedder(Embedder):
    def __init__(self, settings: Settings | None = None):
        settings = settings or Settings()
        self._api_key = settings.GOOGLE_API_KEY
        self._model = settings.GEMINI_EMBEDDING_MODEL
        self._batch_size = settings.EMBEDDING_BATCH_SIZE
        self._client: genai.Client | None = None

    def _ensure_client(self) -> genai.Client:
        if self._client is None:
            if not self._api_key:
                raise ValueError("GOOGLE_API_KEY is not configured")
            self._client = genai.Client(api_key=self._api_key)
        return self._client

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return await self._embed(texts, task_type="RETRIEVAL_DOCUMENT")

    async def embed_query(self, text: str) -> list[float]:
        return (await self._embed([text], task_type="RETRIEVAL_QUERY"))[0]

    async def _embed(self, texts: Sequence[str], task_type: str) -> list[list[float]]:
        if not texts:
            return []
        if any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError("Cannot embed empty text")
        if self._batch_size < 1:
            raise ValueError("EMBEDDING_BATCH_SIZE must be at least 1")

        client = self._ensure_client()
        embeddings: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            response = await asyncio.to_thread(
                client.models.embed_content,
                model=self._model,
                contents=list(texts[start : start + self._batch_size]),
                config=types.EmbedContentConfig(task_type=task_type),
            )
            embeddings.extend(list(item.values) for item in response.embeddings)  # type: ignore

        if len(embeddings) != len(texts):
            raise RuntimeError(
                "Embedding provider returned an unexpected number of vectors"
            )
        return embeddings
