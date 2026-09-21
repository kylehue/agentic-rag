from __future__ import annotations

from collections.abc import Sequence
from typing import Any
from uuid import uuid4

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from app.llm.base import ChatMessage, ContentPart, ToolCall


def to_content(content: str | list[str | dict[str, Any]]) -> str | list[ContentPart]:
    """Narrow langchain message content to the provider wire format.

    The agent's own messages always carry plain strings; list content is
    reduced to its text parts.
    """
    if isinstance(content, str):
        return content
    return [part for part in content if isinstance(part, str)]


def history_text(content: str | list[str | dict[str, Any]]) -> str:
    """The plain text of a message's content (list content is joined)."""
    text = to_content(content)
    if isinstance(text, list):
        return " ".join(part for part in text if isinstance(part, str))
    return text


def to_llm_messages(messages: Sequence[BaseMessage]) -> list[ChatMessage]:
    """Convert langchain messages to the provider-neutral wire format."""
    converted: list[ChatMessage] = []
    for message in messages:
        if isinstance(message, SystemMessage):
            converted.append(
                ChatMessage(role="system", content=to_content(message.content))
            )
        elif isinstance(message, HumanMessage):
            converted.append(
                ChatMessage(role="user", content=to_content(message.content))
            )
        elif isinstance(message, AIMessage):
            # Recover the provider round-trip data the model node stored on
            # the message, keyed by tool call id, and return it to the
            # provider on the next request. The engine does not inspect it.
            provider_data = (message.additional_kwargs or {}).get(
                "tool_call_provider_data"
            ) or {}
            tool_calls: list[ToolCall] = []
            for call in message.tool_calls or ():
                call_id = call["id"] if call["id"] is not None else f"call_{uuid4().hex}"
                tool_calls.append(
                    ToolCall(
                        id=call_id,
                        name=call["name"],
                        arguments=call["args"],
                        provider_data=provider_data.get(call_id),
                    )
                )
            converted.append(
                ChatMessage(
                    role="assistant",
                    content=to_content(message.content),
                    tool_calls=tuple(tool_calls),
                )
            )
        elif isinstance(message, ToolMessage):
            converted.append(
                ChatMessage(
                    role="tool",
                    content=to_content(message.content),
                    tool_call_id=message.tool_call_id,
                )
            )
    return converted
