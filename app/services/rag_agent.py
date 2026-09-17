from __future__ import annotations

import asyncio
import re
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, TypedDict
from uuid import uuid4

import aiosqlite
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import StreamWriter

from app.agent_tools import AgentTool, AgentToolset, flatten_tools, render_tool_blocks
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

# The checkpoint database's file name, inside the agent storage dir.
CHECKPOINT_DB_FILENAME = "checkpoints.sqlite"


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


def _history_text(content: str | list[str | dict[str, Any]]) -> str:
    """The plain text of a message's content (list content is joined)."""
    text = to_content(content)
    if isinstance(text, list):
        return " ".join(part for part in text if isinstance(part, str))
    return text


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
    checkpointer: BaseCheckpointSaver,
):
    """Compile the agent's orchestration graph.

    A two-node loop: the model node asks the LLM (with the tools), and when
    the LLM requests tool calls, the tools node executes them and feeds the
    results back. Once `max_tool_rounds` rounds are used up, the LLM is
    called without tools and must answer, so the loop always terminates.

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
            _to_llm_messages(state["messages"]),
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
    return graph.compile(checkpointer=checkpointer)


def _run_input(
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

{tools}

Rules:
- Answer only from what the tools return. If the evidence is insufficient, say so plainly.
- Keep tool output lean: fetch only what the question needs.
- Cite factual claims as #[origin_source_id:chunk_id].
- Keep the answer concise and direct.
"""


# --- answer citations ---

CHUNK_REF_PATTERN = re.compile(r"#\[([^\[\]:]+):([^\[\]:]+)\]")


def format_chunk_ref(origin_source_id: str, chunk_id: str) -> str:
    """The citation reference for one chunk: `#[origin_source_id:chunk_id]`."""
    return f"#[{origin_source_id}:{chunk_id}]"


def extract_chunk_refs(answer: str) -> dict[str, dict[str, str]]:
    """The chunk references cited in an answer.

    Keyed by the reference string (`#[origin_source_id:chunk_id]`), in
    first-seen order, each mapped to its structured
    `{origin_source_id, chunk_id}` parts. The origin id is the top-level file
    the user uploaded (not the chunk's own source), so citations point at the
    user's documents, including tables embedded in them.
    """
    refs: dict[str, dict[str, str]] = {}
    for match in CHUNK_REF_PATTERN.finditer(answer):
        origin_source_id, chunk_id = match.group(1), match.group(2)
        refs.setdefault(
            format_chunk_ref(origin_source_id, chunk_id),
            {"origin_source_id": origin_source_id, "chunk_id": chunk_id},
        )
    return refs


# --- the service ---


class RagAgentService:
    """Answers questions with a tool-calling agent over the RAG service.

    Takes the RAG service and the tools at initialization (plugin-style).
    Answers are chat-scoped: each `chat_id` is a thread on the graph's
    checkpointer, so a conversation continues across questions and server
    restarts (with a durable checkpointer), and an interrupted run can
    resume. Each run compiles a fresh graph whose tool executors are scoped
    to the chat's chunks. `ask` is a projection of `ask_stream`: one
    execution, two consumers.
    """

    def __init__(
        self,
        *,
        rag_service: RagService,
        tools: Sequence[AgentTool | AgentToolset],
        max_tool_rounds: int = DEFAULT_MAX_TOOL_ROUNDS,
        checkpointer: BaseCheckpointSaver | None = None,
        checkpoint_dir: str | None = None,
    ) -> None:
        # Toolsets wrap their member tools; the model calls the flat tools, so
        # the specs and executors are built from the flattened list with
        # duplicates dropped (the first occurrence of a name wins). The prompt
        # renders the original mix, since that is what carries each toolset's
        # name and instructions.
        seen = set()
        flat_tools: list[AgentTool] = []
        for tool in flatten_tools(tools):
            if tool.name in seen:
                continue
            seen.add(tool.name)
            flat_tools.append(tool)

        self._rag_service = rag_service
        self._tools = flat_tools
        self._max_tool_rounds = max_tool_rounds
        self._specs = [
            ToolSpec(
                name=tool.name,
                description=tool.description,
                parameters=tool.parameters,
            )
            for tool in flat_tools
        ]
        self._system_prompt = render_template(
            RAG_AGENT_SYSTEM_PROMPT_TEMPLATE,
            {"tools": render_tool_blocks(tools)},
        )

        # Checkpoint storage: a provided saver, else a durable SQLite saver
        # created on first use from the agent storage dir (or in-memory when
        # neither is given).
        self._checkpointer = checkpointer
        self._checkpoint_db_path = (
            str(Path(checkpoint_dir) / CHECKPOINT_DB_FILENAME)
            if checkpoint_dir
            else None
        )
        self._checkpoint_conn: aiosqlite.Connection | None = None
        self._checkpointer_lock = asyncio.Lock()
        # The in-flight run per chat (its driving task), so `stop` can
        # interrupt a running answer.
        self._running: dict[str, asyncio.Task] = {}

    async def _ensure_checkpointer(self) -> BaseCheckpointSaver:
        async with self._checkpointer_lock:
            if self._checkpointer is None:
                if self._checkpoint_db_path is None:
                    self._checkpointer = MemorySaver()
                else:
                    Path(self._checkpoint_db_path).parent.mkdir(
                        parents=True, exist_ok=True
                    )
                    self._checkpoint_conn = await aiosqlite.connect(
                        self._checkpoint_db_path
                    )
                    saver = AsyncSqliteSaver(self._checkpoint_conn)
                    # Create the tables now (the saver defers this to the
                    # first write), so reads and deletes work before any
                    # conversation has happened.
                    await saver.setup()
                    self._checkpointer = saver
            return self._checkpointer

    async def initialize(self) -> None:
        """Initialize the services this one wraps (the RAG system tables)
        and open the checkpoint database."""
        await self._rag_service.initialize()
        await self._ensure_checkpointer()

    async def close(self) -> None:
        """Close the checkpoint database connection: the only resource this
        service creates itself. The wrapped RAG service's storages are closed
        by the composition root."""
        if self._checkpoint_conn is not None:
            await self._checkpoint_conn.close()
            self._checkpoint_conn = None
            self._checkpointer = None

    def _create_run(
        self,
        chat_id: str | None,
        checkpointer: BaseCheckpointSaver,
    ):
        """Compile the graph for one run: tool executors scoped to `chat_id`,
        on the shared checkpointer."""
        executors = {
            tool.name: tool.create_executor(self._rag_service, chat_id)
            for tool in self._tools
        }
        return _build_graph(
            self._rag_service.llm,
            self._specs,
            executors,
            self._max_tool_rounds,
            checkpointer,
        )

    async def _prepare_run(
        self,
        question: str,
        chat_id: str,
    ) -> tuple[Any, AgentState | None, RunnableConfig]:
        """The compiled graph, the run input, and the thread config.

        When the thread has a pending step (a run that stopped before
        answering), the input is None and the run resumes from its
        checkpoint; otherwise the question starts a new run on the thread.
        """
        checkpointer = await self._ensure_checkpointer()
        graph = self._create_run(chat_id, checkpointer)
        config: RunnableConfig = {"configurable": {"thread_id": chat_id}}

        state = await graph.aget_state(config)
        if state.next:
            return graph, None, config
        messages = state.values.get("messages") or []
        return graph, _run_input(messages, question, self._system_prompt), config

    async def _run_events(
        self,
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
                    events = _model_events(state)
                else:
                    events = _tool_events(state)
                for event in events:
                    yield event

    def _track_run(self, chat_id: str) -> asyncio.Task | None:
        """Register the current task as the chat's in-flight run (so `stop`
        can interrupt it). Returns the tracked task, or None if there is none."""
        task = asyncio.current_task()
        if task is not None:
            self._running[chat_id] = task
        return task

    def _untrack_run(self, chat_id: str, task: asyncio.Task | None) -> None:
        # Only clear it if it is still this run's task (a newer run for the
        # same chat must not be untracked by an older run's cleanup).
        if task is not None and self._running.get(chat_id) is task:
            del self._running[chat_id]

    async def stop(self, chat_id: str) -> bool:
        """Interrupt the run in flight on a chat, if there is one.

        Cancels the run's task, so the answer is abandoned mid-way; the chat's
        checkpoint keeps whatever step had completed, so the next question can
        resume or start fresh. Returns True when a run was interrupted,
        False when the chat has no run in flight.
        """
        task = self._running.get(chat_id)
        if task is None or task.done():
            return False
        task.cancel()
        return True

    async def ask(
        self,
        question: str,
        *,
        chat_id: str,
    ) -> RagAnswer:
        """Ask a question on a chat's thread and return the answer.

        The chat's prior conversation lives in the thread's checkpointed
        state, and `chat_id` bounds the run's tools to that chat's chunks.
        The answer's `chunk_refs` are the chunks it cites (none, if it cites
        nothing).
        """
        task = self._track_run(chat_id)
        try:
            graph, input_state, config = await self._prepare_run(question, chat_id)
            answer = None
            async for event in self._run_events(graph, input_state, config):
                if isinstance(event, AnswerEvent):
                    answer = event.content
            if answer is None:
                raise RuntimeError("The agent produced no final answer.")
            return RagAnswer(
                query=question,
                answer=answer,
                chunk_refs=extract_chunk_refs(answer),
            )
        finally:
            self._untrack_run(chat_id, task)

    async def ask_stream(
        self,
        question: str,
        *,
        chat_id: str,
    ) -> AsyncIterator[StreamEvent]:
        """Ask a question on a chat's thread and stream the run as it
        happens.

        Yields neutral `StreamEvent`s: `answer_delta` per streamed token,
        `tool_call` per tool the model requests, `tool_result` per result it
        reads back, then the single `answer` event carrying the `RagAnswer`
        (whose `chunk_refs` are the citations in the answer). Raises if the
        run ends without an answer. `chat_id` bounds the run's tools to that
        chat's chunks.
        """
        task = self._track_run(chat_id)
        try:
            graph, input_state, config = await self._prepare_run(question, chat_id)
            # The stream is consumed to completion before the terminal frame
            # is yielded: stopping early would cancel the run and skip its
            # final checkpoint, losing the answer from the chat's history.
            answer = None
            async for event in self._run_events(graph, input_state, config):
                if isinstance(event, AnswerEvent):
                    answer = event.content
                    continue
                item = to_stream_event(event)
                if item is not None:
                    yield item
            if answer is None:
                raise RuntimeError("The agent produced no final answer.")
            yield StreamEvent(
                "answer",
                RagAnswer(
                    query=question,
                    answer=answer,
                    chunk_refs=extract_chunk_refs(answer),
                ),
            )
        finally:
            self._untrack_run(chat_id, task)

    async def chat_history(self, chat_id: str) -> list[dict]:
        """The conversation of a chat, read from the checkpointer, in the same
        order and shape it was streamed: the user's questions, each tool call,
        each tool result, and each answer (with its citations). The system
        prompt and mid-run commentary stay internal. An unknown or empty
        thread yields an empty list.
        """
        # Straight from the checkpointer: the thread's latest checkpoint
        # holds the conversation, no graph or run needed for a read.
        checkpointer = await self._ensure_checkpointer()
        config: RunnableConfig = {"configurable": {"thread_id": chat_id}}
        snapshot = await checkpointer.aget_tuple(config)
        if snapshot is None:
            return []
        values = snapshot.checkpoint.get("channel_values") or {}
        messages = values.get("messages") or []

        history: list[dict] = []
        for message in messages:
            if isinstance(message, HumanMessage):
                history.append(
                    {"type": "user", "content": _history_text(message.content)}
                )
            elif isinstance(message, AIMessage):
                if message.tool_calls:
                    # A tool-requesting turn: expose the calls, not any
                    # mid-run commentary (matching the stream).
                    for call in message.tool_calls:
                        history.append(
                            {
                                "type": "tool_call",
                                "name": call["name"],
                                "arguments": call.get("args") or {},
                            }
                        )
                elif message.content:
                    text = _history_text(message.content)
                    if text:
                        history.append(
                            {
                                "type": "answer",
                                "answer": text,
                                "chunk_refs": extract_chunk_refs(text),
                            }
                        )
            elif isinstance(message, ToolMessage):
                history.append(
                    {
                        "type": "tool_result",
                        "name": message.name,
                        "content": _history_text(message.content),
                    }
                )
        return history

    async def delete_chat(self, chat_id: str) -> None:
        """Delete a chat entirely: its RAG data (files, chunks, vectors) and
        its conversation history (the checkpointer thread)."""
        await self._rag_service.delete_chat(chat_id)
        checkpointer = await self._ensure_checkpointer()
        await checkpointer.adelete_thread(chat_id)


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
