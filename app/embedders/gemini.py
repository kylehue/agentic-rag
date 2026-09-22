import asyncio
from collections.abc import Sequence

from google import genai
from google.genai import types

from app.embedders.base import TextEmbedder


class GeminiEmbedder(TextEmbedder):
    def __init__(self, *, api_key: str, model: str, batch_size: int):
        self._api_key = api_key
        self._model = model
        self._batch_size = batch_size
        self._client: genai.Client | None = None

    def _ensure_client(self) -> genai.Client:
        if self._client is None:
            if not self._api_key:
                raise ValueError("GOOGLE_API_KEY is not configured")
            self._client = genai.Client(api_key=self._api_key)
        return self._client

    async def embed_text(self, texts: Sequence[str]) -> list[list[float]]:
        return await self._embed(texts, task_type="RETRIEVAL_DOCUMENT")

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
            # One Content per text. A list of bare strings is collapsed by the
            # google-genai SDK into a single multi-part Content for the
            # gemini-embedding-2 models, which silently returns one vector.
            contents = [
                types.Content(parts=[types.Part(text=text)])
                for text in texts[start : start + self._batch_size]
            ]
            response = await asyncio.to_thread(
                client.models.embed_content,
                model=self._model,
                # list[Content] is runtime-correct (verified against the API);
                # the stub's invariant union list rejects the narrower list.
                contents=contents,  # type: ignore[arg-type]
                config=types.EmbedContentConfig(task_type=task_type),
            )
            embeddings.extend(list(item.values) for item in response.embeddings)  # type: ignore

        if len(embeddings) != len(texts):
            raise RuntimeError(
                "Embedding provider returned an unexpected number of vectors"
            )
        return embeddings
