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
    role: Literal["user", "assistant"]
    content: str
