import asyncio
from functools import partial

from app.agent.events import AnswerEvent, ToolCallEvent, ToolResultEvent
from app.agent_tools import build_rag_tools
from app.llm.base import RawResult, ToolCall
from app.models.chunk import RetrievedChunk
from app.models.rag import RagAnswer
from app.models.stream import StreamEvent
from app.retrievers.base import Retriever
from app.services.agent_service import AgentService, to_stream_event

from fakes import (
    FakeFileStorage,
    FakeLLM,
    FakeSqlStorage,
    drain,
    tool_call_response,
)


class NoopRetriever(Retriever):
    def __init__(self, chunks: list[RetrievedChunk] | None = None) -> None:
        self._chunks = chunks or []

    async def retrieve(self, user_query) -> list[RetrievedChunk]:
        return self._chunks


def make_chunk(chunk_id: str, plugin: str = "text", text: str = "evidence"):
    return RetrievedChunk(
        chunk_id=chunk_id,
        source_id="s1",
        origin_source_id="s1",
        plugin=plugin,
        text=text,
        score=0.9,
    )


def make_service(llm: FakeLLM, retriever: NoopRetriever) -> AgentService:
    """An AgentService over a fake RAG service (the retriever) and the real
    RAG tools (over fake storages)."""
    return AgentService(
        rag_service=retriever,
        llm=llm,
        tools=partial(
            build_rag_tools,
            sql_storage=FakeSqlStorage(),
            file_storage=FakeFileStorage(),
        ),
    )


# --- runs and evidence ---


def test_create_run_search_records_evidence_deduplicated():
    retriever = NoopRetriever([make_chunk("c1"), make_chunk("c2", "table")])
    service = make_service(FakeLLM(), retriever)
    run = service.create_run()

    async def go():
        await run.agent.tools.execute("search_documents", {"query": "a"})
        await run.agent.tools.execute("search_documents", {"query": "b"})

    asyncio.run(go())

    # The fixed retriever returns both chunks on every search; the run's
    # evidence records them once, in first-seen order.
    assert [chunk.chunk_id for chunk in run.evidence] == ["c1", "c2"]


def test_create_run_yields_fresh_runs_with_isolated_evidence():
    service = make_service(FakeLLM(), NoopRetriever([make_chunk("c1")]))

    run_a, run_b = service.create_run(), service.create_run()
    assert run_a is not run_b
    assert run_a.evidence is not run_b.evidence
    assert run_a.evidence == [] and run_b.evidence == []


# --- stream translation ---


def test_to_stream_event_translates_domain_items():
    assert to_stream_event(ToolCallEvent("t", {"x": 1})) == StreamEvent(
        "tool_call", {"name": "t", "arguments": {"x": 1}}
    )
    assert to_stream_event(ToolResultEvent("t", "out")) == StreamEvent(
        "tool_result", {"name": "t", "content": "out"}
    )
    answer = RagAnswer(query="q", answer="a", chunks=())
    assert to_stream_event(answer) == StreamEvent("answer", answer)
    # The terminal answer event is replaced by the RagAnswer upstream.
    assert to_stream_event(AnswerEvent("a")) is None


# --- system prompt ---


def test_system_prompt_is_built_from_the_tools_outline():
    llm = FakeLLM(RawResult(content="ok"))
    service = make_service(llm, NoopRetriever())

    asyncio.run(service.answer("q"))

    system_prompt = llm.calls[0][0].content
    # The persona and rules...
    assert "retrieval-augmented assistant" in system_prompt
    assert "Cite factual claims" in system_prompt
    # ...and a Tools section derived from the tool definitions.
    assert "Tools:" in system_prompt
    for name in (
        "search_documents",
        "list_tables",
        "inspect_table",
        "query_table",
        "query_chunks",
    ):
        assert f"`{name}`" in system_prompt
    assert "Parameters: query (string, required)" in system_prompt


# --- answer (non-streaming) ---


def test_answer_runs_the_agent_and_reports_its_evidence():
    chunk = make_chunk("c1")
    llm = FakeLLM(
        tool_call_response("search_documents", {"query": "what?"}),
        RawResult(content="Grounded answer."),
    )
    service = make_service(llm, NoopRetriever([chunk]))

    result = asyncio.run(service.answer("what?"))

    assert result.query == "what?"
    # The agent's final answer, with the chunks its search returned as
    # evidence.
    assert result.answer == "Grounded answer."
    assert result.chunks == [chunk]


def test_answer_without_a_search_reports_no_evidence():
    chunk = make_chunk("c1")
    llm = FakeLLM(RawResult(content="No tools needed."))
    service = make_service(llm, NoopRetriever([chunk]))

    result = asyncio.run(service.answer("what?"))

    # The agent answered without searching, so the reported evidence is
    # empty even though the retriever holds chunks.
    assert result.answer == "No tools needed."
    assert result.chunks == []


def test_answer_with_history_passes_the_conversation_through():
    llm = FakeLLM(RawResult(content="Follow-up answer."))
    service = make_service(llm, NoopRetriever())

    result = asyncio.run(
        service.answer(
            "And the south?",
            history=[("user", "North sales?"), ("assistant", "10 units.")],
        )
    )

    assert result.answer == "Follow-up answer."
    conversation = [
        (message.role, message.content)
        for message in llm.calls[0]
        if message.role != "system"
    ]
    assert conversation == [
        ("user", "North sales?"),
        ("assistant", "10 units."),
        ("user", "And the south?"),
    ]


# --- answer_stream ---


def test_answer_stream_yields_steps_then_the_answer():
    llm = FakeLLM(
        tool_call_response("search_documents", {"query": "q"}),
        RawResult(content="the answer"),
    )
    service = make_service(llm, NoopRetriever([make_chunk("c1")]))

    items = asyncio.run(drain(service.answer_stream("q")))

    assert items[0] == StreamEvent(
        "tool_call", {"name": "search_documents", "arguments": {"query": "q"}}
    )
    assert items[1] == StreamEvent(
        "tool_result",
        {
            "name": "search_documents",
            "content": "[1] source=s1 chunk=c1 plugin=text\nevidence",
        },
    )
    assert items[-1].name == "answer"
    final = items[-1].payload
    assert isinstance(final, RagAnswer)
    assert final.answer == "the answer"
    assert [chunk.chunk_id for chunk in final.chunks] == ["c1"]


def test_answer_stream_includes_answer_deltas():
    llm = FakeLLM(RawResult(content="the answer"), chunk_size=2)
    service = make_service(llm, NoopRetriever())

    items = asyncio.run(drain(service.answer_stream("q")))

    names = [item.name for item in items]
    assert names[-1] == "answer"
    assert "answer_delta" in names
    # The deltas carry the tokens and compose the answer...
    deltas = [item for item in items if item.name == "answer_delta"]
    assert "".join(item.payload["content"] for item in deltas) == "the answer"
    assert len(deltas) == 5
    # ...and the terminal answer frame is still canonical.
    assert items[-1].payload.answer == "the answer"


def test_answer_stream_carries_tool_errors_as_steps():
    llm = FakeLLM(
        tool_call_response("nope", {"x": 1}),
        RawResult(content="Recovered."),
    )
    service = make_service(llm, NoopRetriever())

    items = asyncio.run(drain(service.answer_stream("q")))

    error = [
        item
        for item in items
        if item.name == "tool_result" and "unknown tool" in item.payload["content"]
    ]
    assert len(error) == 1
    assert items[-1].name == "answer"
