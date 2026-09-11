from collections.abc import AsyncIterator, Callable, Sequence
from dataclasses import dataclass, field
from typing import Protocol

from app.agent import (
    Agent,
    AgentTools,
    AnswerDeltaEvent,
    AnswerEvent,
    ToolCallEvent,
    ToolResultEvent,
)
from app.agent.graph import DEFAULT_MAX_TOOL_ROUNDS
from app.agent.tools import AgentTool
from app.agent_tools import Retrieve
from app.llm.base import LLMProvider
from app.models.chunk import RetrievedChunk
from app.models.rag import RagAnswer
from app.models.stream import StreamEvent
from app.utils.string import render_template

# The tools section is generated from the tool definitions themselves
# (AgentTools.outline), so the prompt can never list a tool the agent does
# not have, or miss one it does. Cross-tool workflow hints live in the tools'
# own descriptions, not here.
RAG_AGENT_SYSTEM_PROMPT_TEMPLATE = """You are a retrieval-augmented assistant that answers questions about the user's ingested documents.

Work through the tools before answering.

Tools:
{tools}

Rules:
- Answer only from what the tools return. If the evidence is insufficient, say so plainly.
- For tabular data, derive your SQL from the table's description (in the chunk text the search tools return) and its schema (inspect a table before querying it), then compute with SQL instead of estimating from sample rows or reading whole tables.
- Keep tool output lean: fetch only what the question needs.
- Cite factual claims with the source and chunk ids in square brackets, for example [source-id:chunk-id], using the ids the tools show you.
- Keep the answer concise and direct.
"""


class RetrievalProvider(Protocol):
    """The retrieval entry point the agent's search tool runs through (the
    RAG service satisfies it)."""

    async def retrieve(self, user_query: str) -> list[RetrievedChunk]: ...


@dataclass
class AgentRun:
    """A fresh agent plus the evidence collector for one answer.

    Each answer gets its own run so concurrent answers never share evidence.
    """

    agent: Agent
    evidence: list[RetrievedChunk] = field(default_factory=list)


def record_evidence(
    evidence: list[RetrievedChunk], chunks: Sequence[RetrievedChunk]
) -> None:
    """Append the searched chunks to the answer's evidence, deduplicated by
    (source_id, chunk_id), in first-seen order."""
    seen = {(chunk.source_id, chunk.chunk_id) for chunk in evidence}
    for chunk in chunks:
        key = (chunk.source_id, chunk.chunk_id)
        if key in seen:
            continue
        seen.add(key)
        evidence.append(chunk)


class AgentService:
    """Answers questions with an agent over the RAG service.

    Takes the RAG service (its `retrieve` is the agent's search path, plugins'
    finalization included) and the tools — as a builder, so each answer gets
    its own tool set wired to its own evidence-recording retrieval path — and
    composes them into the answering pipeline.
    """

    def __init__(
        self,
        *,
        rag_service: RetrievalProvider,
        llm: LLMProvider,
        tools: Callable[[Retrieve], Sequence[AgentTool]],
        max_tool_rounds: int = DEFAULT_MAX_TOOL_ROUNDS,
    ) -> None:
        self._rag_service = rag_service
        self._llm = llm
        self._tools = tools
        self._max_tool_rounds = max_tool_rounds

    def create_run(self) -> AgentRun:
        """A fresh agent plus evidence collector for one answer.

        The search tool's retrieval path is wrapped so the chunks it returns
        are recorded into this run's `evidence`.
        """
        evidence: list[RetrievedChunk] = []

        async def recording_retrieve(query: str) -> list[RetrievedChunk]:
            chunks = list(await self._rag_service.retrieve(query))
            record_evidence(evidence, chunks)
            return chunks

        tools = AgentTools(self._tools(recording_retrieve))
        agent = Agent(
            self._llm,
            tools,
            system_prompt=render_template(
                RAG_AGENT_SYSTEM_PROMPT_TEMPLATE, {"tools": tools.outline()}
            ),
            max_tool_rounds=self._max_tool_rounds,
        )
        return AgentRun(agent=agent, evidence=evidence)

    async def answer(
        self,
        user_query: str,
        history: Sequence[tuple[str, str]] = (),
    ) -> RagAnswer:
        """Answer through the agent; the reported evidence is the chunks the
        agent's search_documents calls returned (none, if it searched no
        documents). `history` is the prior conversation."""
        run = self.create_run()
        answer = await run.agent.ask(user_query, history=history)
        return RagAnswer(
            query=user_query,
            answer=answer,
            chunks=run.evidence,
        )

    async def answer_stream(
        self,
        user_query: str,
        history: Sequence[tuple[str, str]] = (),
    ) -> AsyncIterator[StreamEvent]:
        """Stream the agent's run for one answer, ending with the answer.

        Yields the run's events as they happen (each token, each tool call,
        each result), then the `answer` event carrying the `RagAnswer` whose
        chunks are the evidence the agent's searches returned. `history` is
        the prior conversation.
        """
        run = self.create_run()
        async for event in run.agent.ask_stream(user_query, history=history):
            if isinstance(event, AnswerEvent):
                yield StreamEvent(
                    "answer",
                    RagAnswer(
                        query=user_query,
                        answer=event.content,
                        chunks=run.evidence,
                    ),
                )
                return
            item = to_stream_event(event)
            if item is not None:
                yield item
        raise RuntimeError("The agent produced no final answer.")


def to_stream_event(item) -> StreamEvent | None:
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
