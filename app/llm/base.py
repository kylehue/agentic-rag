from dataclasses import dataclass
from abc import ABC, abstractmethod
from collections.abc import Sequence


@dataclass(frozen=True)
class LLMAttachment:
    """Binary material supplied with an LLM prompt."""

    content: bytes
    mime_type: str


class LLMProvider(ABC):
    """Provider-neutral interface for generating text from a prompt and attachments."""

    @abstractmethod
    async def answer(
        self,
        query: str,
        attachments: Sequence[LLMAttachment] = (),
    ) -> str:
        """Generate an answer from text and optional provider-neutral attachments."""
