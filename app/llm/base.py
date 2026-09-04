from abc import ABC, abstractmethod


class LLMProvider(ABC):
    """Provider-neutral interface for generating text from a prompt.

    Text-only: this RAG does not send binary attachments to the LLM.
    """

    @abstractmethod
    async def answer(self, query: str) -> str:
        """Generate an answer from a text prompt."""
