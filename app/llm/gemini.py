import asyncio
from collections.abc import Sequence

from google import genai
from google.genai import types
from app.core.config import Settings
from app.llm.base import LLMProvider, LLMAttachment


class GeminiProvider(LLMProvider):
    """Gemini implementation of the LLM interface.

    Flow: answer() > _ensure_client() > Gemini generate_content()
    """

    def __init__(self):
        """Read settings and defer client creation until it is needed."""
        settings = Settings()
        self._api_key = settings.GOOGLE_API_KEY
        self._model = settings.GEMINI_MODEL
        self._client = None

    def _ensure_client(self):
        """Create and reuse the Gemini client, or explain when the API key is missing."""
        if self._client is None:
            if not self._api_key:
                raise ValueError("GOOGLE_API_KEY is not configured")
            self._client = genai.Client(api_key=self._api_key)
        return self._client

    async def answer(
        self, query: str, attachments: Sequence[LLMAttachment] = ()
    ) -> str:
        """Send text and optional binary attachments to Gemini and return its text."""
        client = self._ensure_client()
        contents: str | list[object] = query
        if attachments:
            contents = [
                query,
                *(
                    types.Part.from_bytes(
                        data=attachment.content, mime_type=attachment.mime_type
                    )
                    for attachment in attachments
                ),
            ]
        response = await asyncio.to_thread(
            client.models.generate_content,
            model=self._model,
            contents=contents,  # type: ignore
        )
        return response.text or ""
