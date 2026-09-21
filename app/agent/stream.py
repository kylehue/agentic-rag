from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.runnables import RunnableConfig

from app.agent.events import (
    AgentEvent,
    AnswerDeltaEvent,
    AnswerEvent,
    ToolCallEvent,
    ToolResultEvent,
)
from app.agent.graph import AgentState
from app.agent.messages import to_content


def _update_messages(state: dict) -> list:
    """The node update's messages as a list.

    Nodes return one message or a list; a list here keeps the projection
    uniform. (Iterating a message object directly would silently yield its
    dict entries, which is why this normalizes.)
    """
    messages = state.get("messages") or []
    if not isinstance(messages, list):
        messages = [messages]
    return messages


def model_events(state: dict) -> list[AgentEvent]:
    """The events of one model turn: the tool calls it requested, or its
    answer. A message that carries both is read as tool calls only;
    mid-run commentary is not part of the trace."""
    events: list[AgentEvent] = []
    for message in _update_messages(state):
        if not isinstance(message, AIMessage):
            continue
        if message.tool_calls:
            for call in message.tool_calls:
                events.append(
                    ToolCallEvent(
                        name=call["name"],
                        arguments=call.get("args") or {},
                    )
                )
        elif message.content:
            text = to_content(message.content)
            if isinstance(text, list):
                text = " ".join(part for part in text if isinstance(part, str))
            if text:
                events.append(
                    AnswerEvent(
                        content=text,
                        chunk_refs=message.additional_kwargs.get("citations", {}),
                    )
                )
    return events


def tool_events(state: dict) -> list[AgentEvent]:
    """The events of one tool turn: what each executed tool returned."""
    events: list[AgentEvent] = []
    for message in _update_messages(state):
        if not isinstance(message, ToolMessage):
            continue
        content = message.content
        events.append(
            ToolResultEvent(
                name=message.name or "unknown",
                content=content if isinstance(content, str) else str(content),
            )
        )
    return events


async def run_events(
    graph: Any,
    input_state: AgentState | None,
    config: RunnableConfig,
) -> AsyncIterator[AgentEvent]:
    """Run the graph and project it into the run's event trace."""
    async for mode, data in graph.astream(
        input_state, config, stream_mode=["updates", "custom"]
    ):
        if mode == "custom":
            # The model node's token stream, emitted as it happens.
            if isinstance(data, dict) and data.get("type") == "answer_delta":
                yield AnswerDeltaEvent(content=data["text"])
            continue
        for node, state in data.items():  # type: ignore
            if node == "model":
                events = model_events(state)
            else:
                events = tool_events(state)
            for event in events:
                yield event
