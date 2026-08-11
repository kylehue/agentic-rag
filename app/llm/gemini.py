import asyncio
from typing import Any

from google import genai
from app.core.config import Settings
from app.llm.base import LLMProvider


class GeminiProvider(LLMProvider):
    """Gemini adapter that owns its configuration and creates its client lazily."""

    def __init__(self, client: Any | None = None):
        settings = Settings()
        self._api_key = settings.GOOGLE_API_KEY
        self._model = settings.GEMINI_MODEL
        self._client = client

    async def answer(self, query: str) -> str:
        if self._client is None:
            if not self._api_key:
                raise ValueError("GOOGLE_API_KEY is not configured")
            self._client = genai.Client(api_key=self._api_key)

        response = await asyncio.to_thread(
            self._client.models.generate_content,
            model=self._model,
            contents=query,
        )
        return response.text or ""
