from abc import ABC, abstractmethod


class LLMProvider(ABC):
    @abstractmethod
    async def answer(self, query: str) -> str:
        """Generates an answer from an LLM."""
