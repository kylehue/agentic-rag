from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Annotated, TypedDict

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import StreamWriter

from app.agent.messages import to_llm_messages
from app.agent_tools.base import ToolExecutor
from app.agent_tools.evidence import EvidenceIndex, resolve_citations
from app.llm.base import LLMProvider, RawDelta, ToolSpec, accumulate_result


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    tool_rounds: int


def run_input(
    existing_messages: Sequence[BaseMessage],
    question: str,
    system_prompt: str,
) -> AgentState:
    """The graph input for a new question on a thread: the question (plus
    the system prompt on a fresh thread) and a reset tool budget. The prior
    conversation is already in the thread's checkpointed state."""
    messages: list = []
    if not existing_messages:
        messages.append(SystemMessage(system_prompt))
    messages.append(HumanMessage(question))
    return {
        "messages": messages,
        "tool_rounds": 0,
    }


async def execute_tool(
    executors: dict[str, ToolExecutor],
    name: str,
    arguments: dict,
) -> str:
    """Run one tool call, returning failures to the model as error strings
    (unknown tool or raised exception) so the agent can adapt."""
    executor = executors.get(name)
    if executor is None:
        return f"Error: unknown tool '{name}'."
    try:
        return await executor(arguments)
    except Exception as exc:
        return f"Error in tool '{name}': {exc}"


def build_graph(
    llm: LLMProvider,
    specs: list[ToolSpec],
    executors: dict[str, ToolExecutor],
    max_tool_rounds: int,
    checkpointer: BaseCheckpointSaver,
    evidence: EvidenceIndex,
):
    """Compile the agent's orchestration graph.

    A two-node loop: the model node asks the LLM (with the tools), and when
    the LLM requests tool calls, the tools node executes them and feeds the
    results back. Once `max_tool_rounds` rounds are used up, the LLM is
    called without tools and must answer, so the loop always terminates.

    `evidence` is the run's citation index: the model cites chunks by number
    and the model node resolves those numbers back to real ids on the answer.

    The `checkpointer` persists each thread's state (one thread per
    chat), so a conversation continues across runs and restarts, and an
    interrupted run can resume.
    """

    async def model_node(state: AgentState, writer: StreamWriter) -> dict:
        within_budget = state["tool_rounds"] < max_tool_rounds
        # When the budget is exhausted, `tools` is dropped below, which forces
        # a plain answer. No "answer now" system message is appended, because
        # a system message mid-conversation is rejected by OpenAI ("system
        # message must be at the beginning").

        # Stream the turn: text deltas go out on the custom stream as they
        # happen (token by token), and the accumulated result is what the
        # state gets. A turn that turns out to be tool calls simply has no
        # text deltas (its commentary, if any, is not part of the trace).
        deltas: list[RawDelta] = []
        async for delta in llm.stream_complete(
            to_llm_messages(state["messages"]),
            tools=specs if within_budget else None,
        ):
            deltas.append(delta)
            if delta.text:
                writer({"type": "answer_delta", "text": delta.text})
        result = accumulate_result(deltas)
        # On the answer turn, resolve the citations the model wrote against
        # this run's evidence and attach them to the message, so they persist
        # with it (chat_history reads them back without a live index).
        citations = (
            resolve_citations(result.content, evidence)
            if result.content and not result.tool_calls
            else {}
        )
        # Provider round-trip data for this turn's tool calls, keyed by call
        # id. Stored opaquely on the message so it survives the checkpointed
        # state and is returned to the provider on the next request; the
        # engine does not inspect its contents.
        provider_data = {
            call.id: call.provider_data
            for call in result.tool_calls
            if call.provider_data is not None
        }
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
                    additional_kwargs={
                        "citations": citations,
                        "tool_call_provider_data": provider_data,
                    },
                )
            ]
        }

    async def tools_node(state: AgentState) -> dict:
        last = state["messages"][-1]
        # The route only sends an AIMessage with tool calls here; the guard
        # keeps the state type honest.
        if not isinstance(last, AIMessage):
            return {}

        async def run(call) -> ToolMessage:
            return ToolMessage(
                content=await execute_tool(
                    executors, call["name"], call.get("args") or {}
                ),
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
    return graph.compile(checkpointer=checkpointer)
