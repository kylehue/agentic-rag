import asyncio

from google import genai

from app.core.config import Settings
from app.llm.base import LLMProvider


class GeminiProvider(LLMProvider):
    """Gemini implementation of the LLM interface (text only)."""

    def __init__(self, settings: Settings | None = None):
        """Read settings and defer client creation until it is needed."""
        settings = settings or Settings()
        self._api_key = settings.GOOGLE_API_KEY
        self._model = settings.GEMINI_MODEL
        self._client = None

    def _ensure_client(self) -> genai.Client:
        """Create and reuse the Gemini client, or explain when the API key is missing."""
        if self._client is None:
            if not self._api_key:
                raise ValueError("GOOGLE_API_KEY is not configured")
            self._client = genai.Client(api_key=self._api_key)
        return self._client

    async def answer(self, query: str) -> str:
        """Send a text prompt to Gemini and return its text."""
        client = self._ensure_client()
        response = await asyncio.to_thread(
            client.models.generate_content,
            model=self._model,
            contents=query,
        )
        return response.text or ""
