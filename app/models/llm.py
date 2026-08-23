from dataclasses import dataclass


@dataclass(frozen=True)
class LLMAttachment:
    """Binary material supplied with an LLM prompt."""

    content: bytes
    mime_type: str
