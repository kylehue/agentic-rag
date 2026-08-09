from google import genai

from app.llm.base import LLMProvider

client = genai.Client()


class GeminiProvider(LLMProvider):
    async def answer(self, query):
        response = client.models.generate_content(
            model="gemini-3.5-flash-lite",
            contents=query,
        )
        return response.text or ""
