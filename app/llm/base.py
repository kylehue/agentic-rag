from abc import ABC, abstractmethod
from collections.abc import Sequence

from app.models.llm import LLMAttachment


class LLMProvider(ABC):
    @abstractmethod
    async def answer(
        self, query: str, attachments: Sequence[LLMAttachment] = ()
    ) -> str:
        """Generate an answer from text and optional provider-neutral attachments."""
