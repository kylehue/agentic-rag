from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import aiosqlite
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from app.agent import (
    AgentEvent,
    AgentState,
    AnswerDeltaEvent,
    AnswerEvent,
    ToolCallEvent,
    ToolResultEvent,
    build_graph,
    history_text,
    run_events,
    run_input,
)
from app.agent_tools import AgentTool, AgentToolset, flatten_tools, render_tool_blocks
from app.agent_tools.base import RunContext
from app.agent_tools.evidence import EvidenceIndex
from app.agent_tools.mcp import build_mcp_toolsets
from app.database import agent_table_docs
from app.llm.base import ToolSpec
from app.mcp import McpClient
from app.models.rag import RagAnswer
from app.models.stream import StreamEvent
from app.services.rag import RagService
from app.utils.string import render_template

# The checkpoint database's file name, inside the agent storage dir.
CHECKPOINT_DB_FILENAME = "checkpoints.sqlite"

RAG_AGENT_SYSTEM_PROMPT_TEMPLATE = """You are a retrieval-augmented assistant that answers questions about the user's ingested documents, and nothing else.

{tools}

Records:
{records}

Rules:
- Scope: answer only questions the ingested documents can address. If a question is unrelated to them, say so plainly and do not answer it - no web search, no guessing.
- Web tools: use them only to supplement an answer about the ingested documents (fill in a missing external fact the corpus refers to, or verify/expand on corpus content). Never use them to answer a question that is outside the corpus.
- Answer only from what the tools return. If the evidence is insufficient, say so plainly.
- Keep tool output lean: fetch only what the question needs.
- Cite the evidence you use with the number shown in the tool results, e.g. #[1]. Only do inline citations. Cite only the chunks you actually rely on.
- Keep the answer concise and direct.
"""


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
        max_tool_rounds: int,
        checkpointer: BaseCheckpointSaver | None = None,
        checkpoint_dir: str | None = None,
        mcp_client: McpClient | None = None,
    ) -> None:
        self._rag_service = rag_service
        self._static_tools = list(tools)
        self._mcp_client = mcp_client
        self._max_tool_rounds = max_tool_rounds

        # The static tools define the baseline tool view (specs + prompt);
        # initialize() extends it with the MCP tools once the MCP client has
        # connected.
        self._apply_tools(self._static_tools)

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

    def _apply_tools(self, items: Sequence[AgentTool | AgentToolset]) -> None:
        # Toolsets wrap their member tools; the model calls the flat tools, so
        # the specs and executors are built from the flattened list with
        # duplicates dropped (the first occurrence of a name wins). The prompt
        # renders the original mix, since that is what carries each toolset's
        # name and instructions.
        seen = set()
        flat_tools: list[AgentTool] = []
        for tool in flatten_tools(items):
            if tool.name in seen:
                continue
            seen.add(tool.name)
            flat_tools.append(tool)

        self._tools = flat_tools
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
            {"tools": render_tool_blocks(items), "records": agent_table_docs()},
        )

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
        # Add the MCP tools (the client has connected and cached them), then
        # open the checkpoint database.
        if self._mcp_client is not None:
            all_items: list[AgentTool | AgentToolset] = [
                *self._static_tools,
                *build_mcp_toolsets(self._mcp_client),
            ]
            self._apply_tools(all_items)
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
        context: RunContext,
        checkpointer: BaseCheckpointSaver,
    ):
        """Compile the graph for one run: tool executors bound to the run's
        context, on the shared checkpointer."""
        executors = {
            tool.name: tool.create_executor(self._rag_service, context)
            for tool in self._tools
        }
        return build_graph(
            self._rag_service.llm,
            self._specs,
            executors,
            self._max_tool_rounds,
            checkpointer,
            context.evidence,
        )

    async def _prepare_run(
        self,
        question: str,
        context: RunContext,
    ) -> tuple[Any, AgentState | None, RunnableConfig]:
        """The compiled graph, the run input, and the thread config.

        When the thread has a pending step (a run that stopped before
        answering), the input is None and the run resumes from its
        checkpoint; otherwise the question starts a new run on the thread.
        """
        checkpointer = await self._ensure_checkpointer()
        graph = self._create_run(context, checkpointer)
        config: RunnableConfig = {"configurable": {"thread_id": context.chat_id}}

        state = await graph.aget_state(config)
        if state.next:
            return graph, None, config
        messages = state.values.get("messages") or []
        # The model needs to know "now" to answer time-relative questions. The
        # timestamp is appended to the system prompt, which run_input adds on a
        # fresh thread; a chat keeps the date/time it started with.
        now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
        system_prompt = f"{self._system_prompt}\n\nCurrent date and time: {now}"
        return graph, run_input(messages, question, system_prompt), config

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
        context = RunContext(chat_id=chat_id, query=question, evidence=EvidenceIndex())
        try:
            graph, input_state, config = await self._prepare_run(question, context)
            answer = None
            chunk_refs: dict[str, dict[str, str]] = {}
            async for event in run_events(graph, input_state, config):
                if isinstance(event, AnswerEvent):
                    answer = event.content
                    chunk_refs = event.chunk_refs
            if answer is None:
                raise RuntimeError("The agent produced no final answer.")
            return RagAnswer(
                query=question,
                answer=answer,
                chunk_refs=chunk_refs,
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
        context = RunContext(chat_id=chat_id, query=question, evidence=EvidenceIndex())
        try:
            graph, input_state, config = await self._prepare_run(question, context)
            # The stream is consumed to completion before the terminal frame
            # is yielded: stopping early would cancel the run and skip its
            # final checkpoint, losing the answer from the chat's history.
            answer = None
            chunk_refs: dict[str, dict[str, str]] = {}
            async for event in run_events(graph, input_state, config):
                if isinstance(event, AnswerEvent):
                    answer = event.content
                    chunk_refs = event.chunk_refs
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
                    chunk_refs=chunk_refs,
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
                    {"type": "user", "content": history_text(message.content)}
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
                    text = history_text(message.content)
                    if text:
                        history.append(
                            {
                                "type": "answer",
                                "answer": text,
                                # The model node resolved and stored the
                                # citations on the message, so a refresh reads
                                # them back without a live evidence index.
                                "chunk_refs": message.additional_kwargs.get(
                                    "citations", {}
                                ),
                            }
                        )
            elif isinstance(message, ToolMessage):
                history.append(
                    {
                        "type": "tool_result",
                        "name": message.name,
                        "content": history_text(message.content),
                    }
                )
        return history

    async def delete_chat(self, chat_id: str) -> None:
        """Delete a chat entirely: its RAG data (files, chunks, vectors) and
        its conversation history (the checkpointer thread)."""
        await self._rag_service.delete_chat(chat_id)
        checkpointer = await self._ensure_checkpointer()
        await checkpointer.adelete_thread(chat_id)
