from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel, ConfigDict

from app.api_schemas.chunk import RetrievedChunkSchema

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
    chunks: Sequence[RetrievedChunkSchema]
