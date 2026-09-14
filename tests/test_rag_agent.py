import asyncio

import pytest

from app.agent_tools import (
    InspectTableTool,
    ListTablesTool,
    QueryChunksTool,
    QueryTableTool,
    SearchDocumentTool,
)
from app.llm.base import RawResult, ToolCall
from app.models.chunk import RetrievedChunk
from app.models.rag import RagAnswer
from app.models.stream import StreamEvent
from app.retrievers.base import Retriever
from app.services.rag_agent import (
    AnswerEvent,
    RagAgentService,
    ToolCallEvent,
    ToolResultEvent,
    extract_chunk_refs,
    to_stream_event,
)

from fakes import FakeLLM, build_rag_service, drain, make_tool, tool_call_response

FULL_ANSWER = "The garden plan is in the report."

RAG_TOOLS = [
    SearchDocumentTool(),
    ListTablesTool(),
    InspectTableTool(),
    QueryTableTool(),
    QueryChunksTool(),
]


class NoopRetriever(Retriever):
    def __init__(self, chunks: list[RetrievedChunk] | None = None) -> None:
        self._chunks = chunks or []

    async def retrieve(self, user_query, where=None):
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


def make_agent(llm, tools=None, retriever=None, **kwargs) -> RagAgentService:
    return RagAgentService(
        rag_service=build_rag_service(
            llm, retriever if retriever is not None else NoopRetriever()
        ),
        tools=(
            tools
            if tools is not None
            else [make_tool("search_documents", "found: garden plan")]
        ),
        **kwargs,
    )


# --- ask (non-streaming) ---


def test_ask_uses_a_tool_then_answers():
    calls: list = []
    llm = FakeLLM(
        tool_call_response("search_documents", {"query": "garden"}),
        RawResult(content=FULL_ANSWER),
    )
    agent = make_agent(
        llm, [make_tool("search_documents", "found: garden plan", calls)]
    )

    result = asyncio.run(agent.ask("Where is the garden plan?", chat_id="s1"))

    assert result.answer == FULL_ANSWER
    assert calls == [("search_documents", {"query": "garden"})]
    # The first LLM call was made with the tool spec; the tool result was
    # fed back before the final call.
    assert [spec.name for spec in llm.tools[0]] == ["search_documents"]
    assert any(
        "found: garden plan" in str(message.content) for message in llm.calls[1]
    )


def test_ask_answers_without_tools_when_none_are_requested():
    calls: list = []
    llm = FakeLLM(RawResult(content="Direct answer."))
    agent = make_agent(llm, [make_tool("search_documents", calls=calls)])

    result = asyncio.run(agent.ask("hi", chat_id="s1"))

    assert result.answer == "Direct answer."
    assert calls == []


def test_ask_executes_several_tool_rounds():
    calls: list = []
    llm = FakeLLM(
        tool_call_response("list_tables"),
        tool_call_response("query_table", {"source_id": "tbl-src", "sql": "SELECT 1"}),
        RawResult(content="The total is 35."),
    )
    agent = make_agent(
        llm,
        [
            make_tool("list_tables", "one table: sales", calls),
            make_tool("query_table", "total is 35", calls),
        ],
    )

    result = asyncio.run(agent.ask("total?", chat_id="s1"))

    assert result.answer == "The total is 35."
    assert [name for name, _ in calls] == ["list_tables", "query_table"]


def test_ask_runs_a_round_of_parallel_tool_calls_at_once():
    calls: list = []
    llm = FakeLLM(
        RawResult(
            content="",
            tool_calls=(
                ToolCall(id="c1", name="list_tables", arguments={}),
                ToolCall(id="c2", name="inspect_table", arguments={"source_id": "tbl-src"}),
            ),
        ),
        RawResult(content="Done."),
    )
    agent = make_agent(
        llm,
        [
            make_tool("list_tables", "one table: sales", calls),
            make_tool("inspect_table", "columns: region, amount", calls),
        ],
    )

    result = asyncio.run(agent.ask("describe the sales table", chat_id="s1"))

    assert result.answer == "Done."
    # Both calls of the one round executed...
    assert [name for name, _ in calls] == ["list_tables", "inspect_table"]
    # ...and only two LLM calls happened: the round, then the final answer.
    assert len(llm.calls) == 2
    # Both results were fed back in call order before the final call.
    tool_results = [m for m in llm.calls[1] if m.role == "tool"]
    assert [str(m.content) for m in tool_results] == [
        "one table: sales",
        "columns: region, amount",
    ]
    # The budget was not exhausted, so the final call still offered the
    # tools; the model simply chose to answer.
    assert [spec.name for spec in llm.tools[1]] == ["list_tables", "inspect_table"]


def test_ask_stops_at_the_tool_budget():
    calls: list = []
    llm = FakeLLM(
        tool_call_response("search_documents"),
        RawResult(content="Best effort answer."),
    )
    agent = make_agent(
        llm,
        [make_tool("search_documents", "still searching", calls)],
        max_tool_rounds=1,
    )

    result = asyncio.run(agent.ask("loop?", chat_id="s1"))

    assert result.answer == "Best effort answer."
    # The first round used the tool; the budgeted-out call went to the LLM
    # without tools and produced the final answer.
    assert len(calls) == 1
    assert llm.tools[-1] == []


def test_ask_continues_the_conversation_on_a_chat():
    llm = FakeLLM(
        RawResult(content="First answer."),
        RawResult(content="Second answer."),
    )
    agent = make_agent(llm)

    asyncio.run(agent.ask("tell me about sales", chat_id="s1"))
    result = asyncio.run(agent.ask("now about the south?", chat_id="s1"))

    assert result.answer == "Second answer."
    # The second run's LLM call carries the first exchange from the thread.
    assert [m.role for m in llm.calls[1]] == ["system", "user", "assistant", "user"]


def test_ask_isolated_between_chats():
    llm = FakeLLM(
        RawResult(content="A."),
        RawResult(content="B."),
    )
    agent = make_agent(llm)

    asyncio.run(agent.ask("first", chat_id="s1"))
    asyncio.run(agent.ask("second", chat_id="s2"))

    # The s2 run has no s1 history.
    assert [m.role for m in llm.calls[1]] == ["system", "user"]


def test_ask_resumes_an_interrupted_run():
    # Run 1: the model requests a tool, then the next LLM call crashes.
    llm = FakeLLM(
        tool_call_response("search_documents"),
        RuntimeError("boom"),
        RawResult(content="Recovered."),
    )
    agent = make_agent(llm)

    with pytest.raises(RuntimeError):
        asyncio.run(agent.ask("why?", chat_id="s1"))

    # Asking again resumes the interrupted run from its checkpoint (the
    # question is not appended twice).
    result = asyncio.run(agent.ask("why?", chat_id="s1"))

    assert result.answer == "Recovered."
    final_messages = llm.calls[-1]
    assert [m.role for m in final_messages] == [
        "system",
        "user",
        "assistant",
        "tool",
    ]


def test_ask_tool_errors_are_seen_by_the_model_not_raised():
    llm = FakeLLM(
        tool_call_response("nope", {"x": 1}),
        RawResult(content="Recovered."),
    )
    agent = make_agent(llm, [make_tool("search_documents")])

    result = asyncio.run(agent.ask("q", chat_id="s1"))

    assert result.answer == "Recovered."
    # The unknown tool call came back as an error string for the model.
    feedback = [m for m in llm.calls[1] if "unknown tool" in str(m.content)]
    assert len(feedback) == 1


def test_ask_reports_only_the_cited_chunks():
    # The search returns two chunks, but only one is cited in the answer.
    chunks = [make_chunk("c1"), make_chunk("c2")]
    llm = FakeLLM(
        tool_call_response("search_documents", {"query": "what?"}),
        RawResult(content="Grounded answer. #[s1:c2]"),
    )
    agent = make_agent(llm, [SearchDocumentTool()], retriever=NoopRetriever(chunks))

    result = asyncio.run(agent.ask("what?", chat_id="s1"))

    assert result.query == "what?"
    assert result.answer == "Grounded answer. #[s1:c2]"
    assert result.chunk_refs == {
        "#[s1:c2]": {"source_id": "s1", "chunk_id": "c2"}
    }


def test_ask_without_citations_reports_no_chunk_refs():
    chunk = make_chunk("c1")
    llm = FakeLLM(RawResult(content="No tools needed."))
    agent = make_agent(llm, [SearchDocumentTool()], retriever=NoopRetriever([chunk]))

    result = asyncio.run(agent.ask("what?", chat_id="s1"))

    # The answer cites nothing, so it reports no chunk refs even though the
    # retriever holds chunks.
    assert result.answer == "No tools needed."
    assert result.chunk_refs == {}


def test_ask_raises_when_the_run_never_answers():
    # The fake repeats its last scripted call, so the model keeps requesting
    # the tool even after the budget forces a tool-free final round.
    llm = FakeLLM(tool_call_response("search_documents"))
    agent = make_agent(
        llm,
        [make_tool("search_documents", "still searching")],
        max_tool_rounds=1,
    )

    with pytest.raises(RuntimeError):
        asyncio.run(agent.ask("loop?", chat_id="s1"))


def test_duplicate_tool_names_are_rejected():
    with pytest.raises(ValueError):
        RagAgentService(
            rag_service=build_rag_service(FakeLLM(), NoopRetriever()),
            tools=[make_tool("dup"), make_tool("dup")],
        )


# --- system prompt and stream translation ---


def test_system_prompt_is_built_from_the_tools_outline():
    llm = FakeLLM(RawResult(content="ok"))
    agent = make_agent(llm, RAG_TOOLS)

    asyncio.run(agent.ask("q", chat_id="s1"))

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


def test_to_stream_event_translates_domain_items():
    assert to_stream_event(ToolCallEvent("t", {"x": 1})) == StreamEvent(
        "tool_call", {"name": "t", "arguments": {"x": 1}}
    )
    assert to_stream_event(ToolResultEvent("t", "out")) == StreamEvent(
        "tool_result", {"name": "t", "content": "out"}
    )
    answer = RagAnswer(query="q", answer="a")
    assert to_stream_event(answer) == StreamEvent("answer", answer)
    # The terminal answer event is replaced by the RagAnswer upstream.
    assert to_stream_event(AnswerEvent("a")) is None


def test_extract_chunk_refs_parses_citations():
    answer = "It is here #[s1:c1], again #[s1:c1], and also #[s2:c2]."
    assert extract_chunk_refs(answer) == {
        "#[s1:c1]": {"source_id": "s1", "chunk_id": "c1"},
        "#[s2:c2]": {"source_id": "s2", "chunk_id": "c2"},
    }
    # Plain brackets, ranks, and links are not citations (no leading #).
    assert extract_chunk_refs("[1] source=s1 chunk=c1") == {}
    assert extract_chunk_refs("see [s1:c1] or [a](http://x)") == {}


# --- ask_stream ---


def test_ask_stream_yields_the_run_as_events():
    llm = FakeLLM(
        tool_call_response("search_documents", {"query": "garden"}),
        RawResult(content=FULL_ANSWER),
    )
    agent = make_agent(llm)

    items = asyncio.run(
        drain(agent.ask_stream("Where is the garden plan?", chat_id="s1"))
    )

    # The answer streams as deltas, then the canonical answer terminates the
    # run.
    assert items == [
        StreamEvent(
            "tool_call",
            {"name": "search_documents", "arguments": {"query": "garden"}},
        ),
        StreamEvent(
            "tool_result",
            {"name": "search_documents", "content": "found: garden plan"},
        ),
        StreamEvent("answer_delta", {"content": FULL_ANSWER}),
        StreamEvent(
            "answer",
            # FULL_ANSWER cites nothing, so it carries no chunk refs.
            RagAnswer(query="Where is the garden plan?", answer=FULL_ANSWER),
        ),
    ]


def test_ask_stream_yields_the_answer_token_by_token():
    llm = FakeLLM(
        tool_call_response("search_documents", {"query": "garden"}),
        RawResult(content=FULL_ANSWER),
        chunk_size=4,
    )
    agent = make_agent(llm)

    items = asyncio.run(
        drain(agent.ask_stream("Where is the garden plan?", chat_id="s1"))
    )

    deltas = [item for item in items if item.name == "answer_delta"]
    # The tokens arrive in order and compose the answer...
    assert "".join(item.payload["content"] for item in deltas) == FULL_ANSWER
    assert len(deltas) == (len(FULL_ANSWER) + 3) // 4
    # ...and the terminal answer frame is still canonical.
    assert items[-1].name == "answer"
    assert items[-1].payload.answer == FULL_ANSWER


def test_ask_stream_parallel_calls_yield_every_step_in_order():
    llm = FakeLLM(
        RawResult(
            content="",
            tool_calls=(
                ToolCall(id="c1", name="list_tables", arguments={}),
                ToolCall(id="c2", name="inspect_table", arguments={"source_id": "tbl-src"}),
            ),
        ),
        RawResult(content="Done."),
    )
    agent = make_agent(
        llm,
        [
            make_tool("list_tables", "one table: sales"),
            make_tool("inspect_table", "columns: region, amount"),
        ],
    )

    items = asyncio.run(
        drain(agent.ask_stream("describe the sales table", chat_id="s1"))
    )

    assert items[:5] == [
        StreamEvent("tool_call", {"name": "list_tables", "arguments": {}}),
        StreamEvent(
            "tool_call", {"name": "inspect_table", "arguments": {"source_id": "tbl-src"}}
        ),
        StreamEvent(
            "tool_result", {"name": "list_tables", "content": "one table: sales"}
        ),
        StreamEvent(
            "tool_result",
            {"name": "inspect_table", "content": "columns: region, amount"},
        ),
        StreamEvent("answer_delta", {"content": "Done."}),
    ]
    assert items[-1].name == "answer"


def test_ask_stream_streams_tool_turn_commentary_before_the_call():
    # A turn that carries both text and tool calls streams its text as
    # deltas, then reports the call: the commentary is visible, but it is
    # not the answer.
    llm = FakeLLM(
        RawResult(
            content="checking now",
            tool_calls=(ToolCall(id="c1", name="search_documents", arguments={}),),
        ),
        RawResult(content="Done."),
        chunk_size=3,
    )
    agent = make_agent(llm, [make_tool("search_documents", "found")])

    items = asyncio.run(drain(agent.ask_stream("q", chat_id="s1")))

    assert items[:4] == [
        StreamEvent("answer_delta", {"content": "che"}),
        StreamEvent("answer_delta", {"content": "cki"}),
        StreamEvent("answer_delta", {"content": "ng "}),
        StreamEvent("answer_delta", {"content": "now"}),
    ]
    assert items[4] == StreamEvent(
        "tool_call", {"name": "search_documents", "arguments": {}}
    )
    assert items[5] == StreamEvent(
        "tool_result", {"name": "search_documents", "content": "found"}
    )
    assert items[-1].name == "answer"
    assert items[-1].payload.answer == "Done."


def test_ask_stream_carries_tool_errors_as_steps():
    llm = FakeLLM(
        tool_call_response("nope", {"x": 1}),
        RawResult(content="Recovered."),
    )
    agent = make_agent(llm, [make_tool("search_documents")])

    items = asyncio.run(drain(agent.ask_stream("q", chat_id="s1")))

    error = [
        item
        for item in items
        if item.name == "tool_result" and "unknown tool" in item.payload["content"]
    ]
    assert len(error) == 1
    assert items[-1].name == "answer"


def test_ask_stream_raises_when_the_run_never_answers():
    llm = FakeLLM(tool_call_response("search_documents"))
    agent = make_agent(
        llm,
        [make_tool("search_documents", "still searching")],
        max_tool_rounds=1,
    )

    async def go():
        await drain(agent.ask_stream("loop?", chat_id="s1"))

    with pytest.raises(RuntimeError):
        asyncio.run(go())
