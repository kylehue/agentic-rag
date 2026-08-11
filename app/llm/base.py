from abc import ABC, abstractmethod


class LLMProvider(ABC):
    @abstractmethod
    async def answer(self, query: str) -> str:
        """Generates an answer from an LLM."""

    @abstractmethod
    async def answer_image(
        self,
        query: str,
        image: bytes,
        mime_type: str,
    ) -> str:
        """Describes an image with optional textual context."""
        raise NotImplementedError("This LLM provider does not support image input")
