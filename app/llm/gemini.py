import asyncio
from typing import Any

from google import genai
from google.genai import types
from app.core.config import Settings
from app.llm.base import LLMProvider


class GeminiProvider(LLMProvider):
    """Gemini adapter that owns its configuration and creates its client lazily."""

    def __init__(self):
        settings = Settings()
        self._api_key = settings.GOOGLE_API_KEY
        self._model = settings.GEMINI_MODEL
        self._client = None

    def _ensure_client(self):
        if self._client is None:
            if not self._api_key:
                raise ValueError("GOOGLE_API_KEY is not configured")
            self._client = genai.Client(api_key=self._api_key)
        return self._client

    async def answer(self, query: str) -> str:
        client = self._ensure_client()

        response = await asyncio.to_thread(
            client.models.generate_content,
            model=self._model,
            contents=query,
        )
        return response.text or ""

    async def answer_image(self, query: str, image: bytes, mime_type: str) -> str:
        client = self._ensure_client()

        response = await asyncio.to_thread(
            client.models.generate_content,
            model=self._model,
            contents=[
                query,
                types.Part.from_bytes(data=image, mime_type=mime_type),
            ],
        )
        return response.text or ""
