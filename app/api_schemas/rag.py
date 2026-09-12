from typing import Literal

from pydantic import BaseModel, ConfigDict

HistoryRole = Literal["user", "assistant"]


class HistoryMessageSchema(BaseModel):
    """One prior turn of the conversation."""

    role: HistoryRole
    content: str


class RagAnswerRequestSchema(BaseModel):
    """The question, plus the prior conversation (oldest first) for
    multi-turn answers."""

    query: str
    history: list[HistoryMessageSchema] = []


class RagAnswerSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    query: str
    answer: str
    chunk_refs: dict[str, dict[str, str]]
