from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Annotated, Any, TypedDict
from uuid import uuid4

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolCall as LangChainToolCall,
    ToolMessage,
)
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import StreamWriter

from app.agent.tools import AgentTools
from app.llm.base import (
    ChatMessage,
    ContentPart,
    LLMProvider,
    RawDelta,
    ToolCall,
    accumulate_result,
)

DEFAULT_MAX_TOOL_ROUNDS = 8


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    tool_rounds: int


def to_content(content: str | list[str | dict[str, Any]]) -> str | list[ContentPart]:
    """Narrow langchain message content to the provider wire format.

    The agent's own messages always carry plain strings; list content is
    reduced to its text parts.
    """
    if isinstance(content, str):
        return content
    return [part for part in content if isinstance(part, str)]


def _to_llm_messages(messages: Sequence[BaseMessage]) -> list[ChatMessage]:
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
            converted.append(
                ChatMessage(
                    role="assistant",
                    content=to_content(message.content),
                    tool_calls=tuple(
                        ToolCall(
                            id=(
                                call["id"]
                                if call["id"] is not None
                                else f"call_{uuid4().hex}"
                            ),
                            name=call["name"],
                            arguments=call["args"],
                        )
                        for call in (message.tool_calls or ())
                    ),
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


def build_agent(
    llm: LLMProvider,
    tools: AgentTools,
    *,
    max_tool_rounds: int = DEFAULT_MAX_TOOL_ROUNDS,
):
    """Compile the agent's orchestration graph.

    A two-node loop: the model node asks the LLM (with the tools), and when
    the LLM requests tool calls, the tools node executes them and feeds the
    results back. Once `max_tool_rounds` rounds are used up, the LLM is
    called without tools and must answer, so the loop always terminates.
    """
    specs = tools.specs()

    async def model_node(state: AgentState, writer: StreamWriter) -> dict:
        within_budget = state["tool_rounds"] < max_tool_rounds
        messages = list(state["messages"])
        if not within_budget:
            messages.append(
                SystemMessage(
                    content=(
                        "The tool budget is exhausted. Answer now with the "
                        "information you have."
                    )
                )
            )

        # Stream the turn: text deltas go out on the custom stream as they
        # happen (token by token), and the accumulated result is what the
        # state gets. A turn that turns out to be tool calls simply has no
        # text deltas (its commentary, if any, is not part of the trace).
        deltas: list[RawDelta] = []
        async for delta in llm.stream_complete(
            _to_llm_messages(messages),
            tools=specs if within_budget else None,
        ):
            deltas.append(delta)
            if delta.text:
                writer({"type": "answer_delta", "text": delta.text})
        result = accumulate_result(deltas)
        # A list, like the tools node: node outputs are the unit the state
        # (and any stream projection) sees.
        return {
            "messages": [
                AIMessage(
                    content=result.content or "",
                    tool_calls=[
                        {"name": call.name, "args": call.arguments, "id": call.id}
                        for call in result.tool_calls
                    ],
                )
            ]
        }

    async def tools_node(state: AgentState) -> dict:
        last = state["messages"][-1]
        # The route only sends an AIMessage with tool calls here; the guard
        # keeps the state type honest.
        if not isinstance(last, AIMessage):
            return {}

        async def run(call: LangChainToolCall) -> ToolMessage:
            return ToolMessage(
                content=await tools.execute(call["name"], call.get("args") or {}),
                tool_call_id=call["id"],
                name=call["name"],
            )

        # The tools are read-only and independent, so a round of parallel
        # calls runs concurrently; gather preserves the model's call order.
        outputs = list(await asyncio.gather(*(run(call) for call in last.tool_calls)))
        return {"messages": outputs, "tool_rounds": state["tool_rounds"] + 1}

    def route(state: AgentState) -> str:
        last = state["messages"][-1]
        if (
            isinstance(last, AIMessage)
            and last.tool_calls
            and state["tool_rounds"] < max_tool_rounds
        ):
            return "tools"
        return END

    graph = StateGraph(AgentState)
    graph.add_node("model", model_node)
    graph.add_node("tools", tools_node)
    graph.add_edge(START, "model")
    graph.add_conditional_edges("model", route, {"tools": "tools", END: END})
    graph.add_edge("tools", "model")
    return graph.compile()
