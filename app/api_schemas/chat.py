from typing import Literal

from pydantic import BaseModel


class ChatInfo(BaseModel):
    chat_id: str
    created_at: float
    # The origin source ids ingested into this chat.
    sources: list[str] = []


class ChatCreated(BaseModel):
    chat_id: str


class ChatMessage(BaseModel):
    """One item in a chat's history, in the order it happened.

    Mirrors the discrete events of the answer stream (plus the user's
    questions), so a client can re-render a refreshed conversation the same
    way it rendered the live stream. Each item carries only the fields for
    its `type`.
    """

    type: Literal["user", "tool_call", "tool_result", "answer"]
    content: str | None = None  # user: the question; tool_result: the output
    name: str | None = None  # tool_call / tool_result: the tool name
    arguments: dict | None = None  # tool_call: the arguments
    answer: str | None = None  # answer: the answer text
    chunk_refs: dict[str, dict[str, str]] | None = None  # answer: the citations
