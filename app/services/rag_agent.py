from __future__ import annotations

import asyncio
import re
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from typing import Annotated, Any, TypedDict
from uuid import uuid4

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import StreamWriter

from app.agent_tools import AgentTool, tools_outline
from app.agent_tools.base import ToolExecutor
from app.llm.base import (
    ChatMessage,
    ContentPart,
    LLMProvider,
    RawDelta,
    ToolCall,
    ToolSpec,
    accumulate_result,
)
from app.models.rag import RagAnswer
from app.models.stream import StreamEvent
from app.services.rag import RagService
from app.utils.string import render_template

DEFAULT_MAX_TOOL_ROUNDS = 8


# --- the run's event trace ---


@dataclass(frozen=True)
class AgentEvent:
    """One observable step of an agent run."""


@dataclass(frozen=True)
class AnswerDeltaEvent(AgentEvent):
    """An increment of the model's text as the current turn streams.

    An advisory preview of the in-flight turn: if the turn ends with tool
    calls, its deltas were commentary (the trace carries only the tool
    call); if the turn is the answer, they compose it. The final
    `AnswerEvent` remains the canonical text.
    """

    content: str


@dataclass(frozen=True)
class ToolCallEvent(AgentEvent):
    """The model asked to call a tool (one per call, in the model's order)."""

    name: str
    arguments: dict


@dataclass(frozen=True)
class ToolResultEvent(AgentEvent):
    """A tool ran; `content` is what the model reads back (result or error)."""

    name: str
    content: str


@dataclass(frozen=True)
class AnswerEvent(AgentEvent):
    """The agent's final answer text. The last event of a run."""

    content: str


# --- the orchestration graph ---


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


def _build_graph(
    llm: LLMProvider,
    specs: list[ToolSpec],
    executors: dict[str, ToolExecutor],
    max_tool_rounds: int,
):
    """Compile the agent's orchestration graph.

    A two-node loop: the model node asks the LLM (with the tools), and when
    the LLM requests tool calls, the tools node executes them and feeds the
    results back. Once `max_tool_rounds` rounds are used up, the LLM is
    called without tools and must answer, so the loop always terminates.
    """

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
    return graph.compile()


def _initial_state(
    question: str,
    history: Sequence[tuple[str, str]],
    system_prompt: str,
) -> AgentState:
    """The graph's starting state for one question.

    `history` is the prior conversation, oldest first, as (role, content)
    pairs with role "user" or "assistant". It is placed between the system
    prompt and the current question, so the agent can answer follow-ups.
    """
    messages: list = [SystemMessage(system_prompt)]
    for role, content in history:
        if role == "user":
            messages.append(HumanMessage(content))
        else:
            messages.append(AIMessage(content))
    messages.append(HumanMessage(question))
    return {
        "messages": messages,
        "tool_rounds": 0,
    }


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


def _model_events(state: dict) -> list[AgentEvent]:
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
                events.append(AnswerEvent(content=text))
    return events


def _tool_events(state: dict) -> list[AgentEvent]:
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


# --- the system prompt and per-answer runs ---

RAG_AGENT_SYSTEM_PROMPT_TEMPLATE = """You are a retrieval-augmented assistant that answers questions about the user's ingested documents.

Work through the tools before answering.

Tools:
{tools}

Rules:
- Answer only from what the tools return. If the evidence is insufficient, say so plainly.
- For tabular data, derive your SQL from the table's description (in the chunk text the search tools return) and its schema (inspect a table's schema before querying it), then compute with SQL instead of estimating from sample rows or reading whole tables.
- Keep tool output lean: fetch only what the question needs.
- Cite factual claims as #[source_id:chunk_id].
- Keep the answer concise and direct.
"""


# --- answer citations ---

CHUNK_REF_PATTERN = re.compile(r"#\[([^\[\]:]+):([^\[\]:]+)\]")


def format_chunk_ref(source_id: str, chunk_id: str) -> str:
    """The citation reference for one chunk: `#[source_id:chunk_id]`."""
    return f"#[{source_id}:{chunk_id}]"


def extract_chunk_refs(answer: str) -> dict[str, dict[str, str]]:
    """The chunk references cited in an answer.

    Keyed by the reference string (`#[source_id:chunk_id]`), in first-seen
    order, each mapped to its structured `{source_id, chunk_id}` parts.
    """
    refs: dict[str, dict[str, str]] = {}
    for match in CHUNK_REF_PATTERN.finditer(answer):
        source_id, chunk_id = match.group(1), match.group(2)
        refs.setdefault(
            format_chunk_ref(source_id, chunk_id),
            {"source_id": source_id, "chunk_id": chunk_id},
        )
    return refs


# --- the service ---


class RagAgentService:
    """Answers questions with a tool-calling agent over the RAG service."""

    def __init__(
        self,
        *,
        rag_service: RagService,
        tools: Sequence[AgentTool],
        max_tool_rounds: int = DEFAULT_MAX_TOOL_ROUNDS,
    ) -> None:
        seen = set()
        for tool in tools:
            if tool.name in seen:
                raise ValueError(f"Tool name '{tool.name}' is already registered.")
            seen.add(tool.name)

        self._rag_service = rag_service
        self._tools = list(tools)
        self._max_tool_rounds = max_tool_rounds
        self._specs = [
            ToolSpec(
                name=tool.name,
                description=tool.description,
                parameters=tool.parameters,
            )
            for tool in tools
        ]
        self._system_prompt = render_template(
            RAG_AGENT_SYSTEM_PROMPT_TEMPLATE, {"tools": tools_outline(tools)}
        )

    def _create_run(self):
        """Compile a fresh graph for one answer, with per-answer tool
        executors."""
        executors = {
            tool.name: tool.create_executor(self._rag_service) for tool in self._tools
        }
        return _build_graph(
            self._rag_service.llm,
            self._specs,
            executors,
            self._max_tool_rounds,
        )

    async def _run_events(
        self,
        graph: Any,
        question: str,
        history: Sequence[tuple[str, str]],
    ) -> AsyncIterator[AgentEvent]:
        """Run the graph and project it into the run's event trace."""
        async for mode, data in graph.astream(
            _initial_state(question, history, self._system_prompt),
            stream_mode=["updates", "custom"],
        ):
            if mode == "custom":
                # The model node's token stream, emitted as it happens.
                if isinstance(data, dict) and data.get("type") == "answer_delta":
                    yield AnswerDeltaEvent(content=data["text"])
                continue
            for node, state in data.items():  # type: ignore
                if node == "model":
                    events = _model_events(state)
                else:
                    events = _tool_events(state)
                for event in events:
                    yield event

    async def ask(
        self,
        question: str,
        history: Sequence[tuple[str, str]] = (),
    ) -> RagAnswer:
        """Ask a question and return the answer.

        The answer's `chunk_refs` are the chunks it cites (parsed from the
        `#[source_id:chunk_id]` references in the answer text; none, if it
        cites nothing). `history` is the prior conversation.
        """
        graph = self._create_run()
        answer = None
        async for event in self._run_events(graph, question, history):
            if isinstance(event, AnswerEvent):
                answer = event.content
        if answer is None:
            raise RuntimeError("The agent produced no final answer.")
        return RagAnswer(
            query=question,
            answer=answer,
            chunk_refs=extract_chunk_refs(answer),
        )

    async def ask_stream(
        self,
        question: str,
        history: Sequence[tuple[str, str]] = (),
    ) -> AsyncIterator[StreamEvent]:
        """Ask a question and stream the run as it happens.

        Yields neutral `StreamEvent`s: `answer_delta` per streamed token,
        `tool_call` per tool the model requests, `tool_result` per result it
        reads back, then the single `answer` event carrying the `RagAnswer`
        (whose `chunk_refs` are the citations in the answer). Raises if the
        run ends without an answer. `history` is the prior conversation.
        """
        graph = self._create_run()
        async for event in self._run_events(graph, question, history):
            if isinstance(event, AnswerEvent):
                yield StreamEvent(
                    "answer",
                    RagAnswer(
                        query=question,
                        answer=event.content,
                        chunk_refs=extract_chunk_refs(event.content),
                    ),
                )
                return
            item = to_stream_event(event)
            if item is not None:
                yield item
        raise RuntimeError("The agent produced no final answer.")


def to_stream_event(item: AgentEvent | RagAnswer) -> StreamEvent | None:
    """Translate one domain stream item into a neutral `StreamEvent`.

    Step events become `StreamEvent`s with dict payloads; the terminal
    `RagAnswer` becomes the `answer` event (its payload stays the object for
    the API to schema-serialize); anything else is dropped.
    """
    if isinstance(item, ToolCallEvent):
        return StreamEvent(
            "tool_call", {"name": item.name, "arguments": item.arguments}
        )
    if isinstance(item, ToolResultEvent):
        return StreamEvent("tool_result", {"name": item.name, "content": item.content})
    if isinstance(item, AnswerDeltaEvent):
        return StreamEvent("answer_delta", {"content": item.content})
    if isinstance(item, RagAnswer):
        return StreamEvent("answer", item)
    return None
