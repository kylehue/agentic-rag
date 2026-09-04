import asyncio

from openai import OpenAI

from app.core.config import Settings
from app.llm.base import LLMProvider


class OpenAIProvider(LLMProvider):
    """OpenAI implementation of the LLM interface (text only)."""

    def __init__(self, settings: Settings | None = None):
        """Read settings and defer client creation until it is needed."""
        settings = settings or Settings()
        self._api_key = settings.OPENAI_API_KEY
        self._model = settings.OPENAI_MODEL
        self._base_url = settings.OPENAI_BASE_URL
        self._client = None

    def _ensure_client(self) -> OpenAI:
        """Create and reuse the OpenAI client, or explain when the API key is missing."""
        if self._client is None:
            if not self._api_key:
                raise ValueError("OPENAI_API_KEY is not configured")
            self._client = OpenAI(
                api_key=self._api_key,
                base_url=self._base_url,
            )
        return self._client

    async def answer(self, query: str) -> str:
        """Send a text prompt to OpenAI and return its text."""
        client = self._ensure_client()
        response = await asyncio.to_thread(
            client.chat.completions.create,
            model=self._model,
            messages=[{"role": "user", "content": query}],
        )
        return response.choices[0].message.content or ""
