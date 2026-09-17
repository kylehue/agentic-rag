import asyncio

import pytest

from app.agent_tools import (
    AgentToolset,
    InspectTableRelationshipsTool,
    InspectTableTool,
    SearchDocumentTool,
    SqlQueryDocumentsTool,
    SqlQueryTableTool,
    RAG_TOOLSET,
    flatten_tools,
    render_tool_blocks,
)
from app.llm.base import RawDelta, RawResult, ToolCall
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
    InspectTableTool(),
    InspectTableRelationshipsTool(),
    SqlQueryTableTool(),
    SqlQueryDocumentsTool(),
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
        tool_call_response("inspect_table", {"source_id": "tbl-src"}),
        tool_call_response("sql_query_table", {"source_id": "tbl-src", "sql": "SELECT 1"}),
        RawResult(content="The total is 35."),
    )
    agent = make_agent(
        llm,
        [
            make_tool("inspect_table", "columns: region, amount", calls),
            make_tool("sql_query_table", "total is 35", calls),
        ],
    )

    result = asyncio.run(agent.ask("total?", chat_id="s1"))

    assert result.answer == "The total is 35."
    assert [name for name, _ in calls] == ["inspect_table", "sql_query_table"]


def test_ask_runs_a_round_of_parallel_tool_calls_at_once():
    calls: list = []
    llm = FakeLLM(
        RawResult(
            content="",
            tool_calls=(
                ToolCall(id="c1", name="inspect_table", arguments={"source_id": "tbl-src"}),
                ToolCall(id="c2", name="inspect_table_relationships", arguments={"source_id": "tbl-src"}),
            ),
        ),
        RawResult(content="Done."),
    )
    agent = make_agent(
        llm,
        [
            make_tool("inspect_table", "columns: region, amount", calls),
            make_tool("inspect_table_relationships", "no siblings", calls),
        ],
    )

    result = asyncio.run(agent.ask("describe the sales table", chat_id="s1"))

    assert result.answer == "Done."
    # Both calls of the one round executed...
    assert [name for name, _ in calls] == [
        "inspect_table",
        "inspect_table_relationships",
    ]
    # ...and only two LLM calls happened: the round, then the final answer.
    assert len(llm.calls) == 2
    # Both results were fed back in call order before the final call.
    tool_results = [m for m in llm.calls[1] if m.role == "tool"]
    assert [str(m.content) for m in tool_results] == [
        "columns: region, amount",
        "no siblings",
    ]
    # The budget was not exhausted, so the final call still offered the
    # tools; the model simply chose to answer.
    assert [spec.name for spec in llm.tools[1]] == [
        "inspect_table",
        "inspect_table_relationships",
    ]


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


def test_stop_interrupts_a_running_answer():
    # An LLM that blocks on its first call, so the run is in flight and can
    # be interrupted.
    class BlockingLLM(FakeLLM):
        def __init__(self) -> None:
            super().__init__("never")
            self.entered = asyncio.Event()

        async def stream_complete(self, messages, *, tools=None, json_schema=None):
            self.entered.set()
            await asyncio.Event().wait()  # block until the task is cancelled
            yield RawDelta(text="unreachable")

    llm = BlockingLLM()
    agent = make_agent(llm)

    async def flow():
        await agent.initialize()
        task = asyncio.create_task(agent.ask("hi", chat_id="s1"))
        await asyncio.wait_for(llm.entered.wait(), timeout=2)
        stopped = await agent.stop("s1")
        with pytest.raises(asyncio.CancelledError):
            await task
        return stopped

    assert asyncio.run(flow()) is True


def test_stop_returns_false_when_nothing_is_running():
    agent = make_agent(FakeLLM("ok"))

    async def flow():
        await agent.initialize()
        return await agent.stop("s1")

    assert asyncio.run(flow()) is False


def test_delete_chat_removes_the_history_and_the_rag_data(tmp_path):
    from app.core.config import settings
    from app.services.rag import RagService
    from app.store_file.local import LocalFileStorage
    from app.store_sql.local import LocalSqlStorage

    from fakes import FakeEmbedder, FakeVectorStorage
    from test_file_management import seed_tree

    sql_storage = LocalSqlStorage(storage_dir=tmp_path / "sql")
    file_storage = LocalFileStorage(storage_dir=tmp_path / "file")
    vector_storage = FakeVectorStorage()
    rag = RagService(
        llm=FakeLLM("ok"),
        embedder=FakeEmbedder(),
        retriever=NoopRetriever(),
        vector_storage=vector_storage,
        sql_storage=sql_storage,
        file_storage=file_storage,
    )
    agent = RagAgentService(rag_service=rag, tools=[make_tool("search_documents")])

    async def flow():
        await rag.initialize()
        await seed_tree(sql_storage, file_storage, vector_storage)
        # A conversation on chat-1, so the checkpointer has a thread.
        await agent.ask("hi", chat_id="chat-1")
        before = await agent.chat_history("chat-1")
        await agent.delete_chat("chat-1")
        after = await agent.chat_history("chat-1")
        chunk_rows = await sql_storage.get_all(
            settings.CHUNK_TABLE_NAME, condition=lambda t: t.c.chat_id == "chat-1"
        )
        doc_rows = await sql_storage.get_all(
            settings.DOCUMENT_METADATA_TABLE_NAME,
            condition=lambda t: t.c.chat_id == "chat-1",
        )
        await sql_storage.close()
        return before, after, chunk_rows, doc_rows

    before, after, chunk_rows, doc_rows = asyncio.run(flow())

    # The conversation existed (a question and an answer), then the thread
    # was deleted.
    assert [item["type"] for item in before] == ["user", "answer"]
    assert after == []
    # The chat's RAG data is gone too.
    assert chunk_rows == []
    assert doc_rows == []
    # Its vectors were removed.
    assert sorted(vector_storage.deleted[0]) == ["c1", "c2"]


def test_delete_chat_with_no_history_on_a_durable_checkpointer(tmp_path):
    # A SQLite checkpointer creates its tables lazily; deleting a chat that
    # never had a conversation must not fail on a missing table.
    agent = RagAgentService(
        rag_service=build_rag_service(FakeLLM("ok"), NoopRetriever()),
        tools=[make_tool("search_documents")],
        checkpoint_dir=str(tmp_path / "ckpts"),
    )

    async def flow():
        await agent.initialize()
        await agent.delete_chat("never-asked")
        assert await agent.chat_history("never-asked") == []

    asyncio.run(flow())


def test_chat_history_includes_tool_traffic_in_stream_order():
    llm = FakeLLM(
        tool_call_response("search_documents", {"query": "what?"}),
        RawResult(content="The answer."),
    )
    agent = make_agent(llm, [make_tool("search_documents", "evidence found")])

    async def flow():
        await agent.ask("q", chat_id="s1")
        return await agent.chat_history("s1")

    history = asyncio.run(flow())

    # The history mirrors the stream: question, tool call, tool result, answer.
    assert [item["type"] for item in history] == [
        "user",
        "tool_call",
        "tool_result",
        "answer",
    ]
    assert history[0] == {"type": "user", "content": "q"}
    assert history[1] == {
        "type": "tool_call",
        "name": "search_documents",
        "arguments": {"query": "what?"},
    }
    assert history[2] == {
        "type": "tool_result",
        "name": "search_documents",
        "content": "evidence found",
    }
    assert history[3]["answer"] == "The answer."
    assert history[3]["chunk_refs"] == {}


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
        "#[s1:c2]": {"origin_source_id": "s1", "chunk_id": "c2"}
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


def test_duplicate_bare_tool_names_are_deduped():
    agent = RagAgentService(
        rag_service=build_rag_service(FakeLLM(), NoopRetriever()),
        tools=[make_tool("dup"), make_tool("dup")],
    )
    # The first occurrence wins; the duplicate is dropped, not an error.
    assert [tool.name for tool in agent._tools] == ["dup"]
    assert [spec.name for spec in agent._specs] == ["dup"]


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
        "inspect_table",
        "inspect_table_relationships",
        "sql_query_table",
        "sql_query_documents",
    ):
        assert f"`{name}`" in system_prompt
    assert "Parameters: query (string, required)" in system_prompt


def test_toolset_outline_lists_each_tool_and_render_adds_instructions():
    toolset = AgentToolset(
        name="Test set",
        tools=[make_tool("alpha"), make_tool("beta")],
        instructions="Call alpha before beta.",
    )

    # outline() is just the member tools' specs...
    assert "`alpha`" in toolset.outline()
    assert "`beta`" in toolset.outline()
    assert "Call alpha" not in toolset.outline()
    # ...and render() brings the set's name and instructions together with them.
    assert toolset.render() == (
        "Toolset name: Test set\n\n"
        "Toolset Description:\n"
        "Call alpha before beta.\n\n"
        "Tools:\n" + toolset.outline()
    )


def test_flatten_tools_unwraps_toolsets_and_render_tool_blocks_groups_bare():
    toolset = AgentToolset(
        name="Set", tools=[make_tool("alpha")], instructions="Instr."
    )
    beta = make_tool("beta")

    # flatten_tools returns the bare tools in order, unwrapping toolsets.
    assert [t.name for t in flatten_tools([toolset, beta])] == ["alpha", "beta"]
    # render_tool_blocks: the toolset's block (name + instructions + specs)
    # plus a grouped Tools: header for the bare tool.
    blocks = render_tool_blocks([toolset, beta])
    assert "Set" in blocks
    assert "Instr." in blocks
    assert "`alpha`" in blocks
    assert "`beta`" in blocks
    assert blocks.count("Tools:") == 2


def test_agent_accepts_a_toolset_and_uses_its_instructions():
    llm = FakeLLM(RawResult(content="ok"))
    toolset = AgentToolset(
        name="Search set",
        tools=[make_tool("search_documents")],
        instructions="Always search first.",
    )
    agent = RagAgentService(
        rag_service=build_rag_service(llm, NoopRetriever()),
        tools=[toolset],
    )

    asyncio.run(agent.ask("q", chat_id="s1"))

    system_prompt = llm.calls[0][0].content
    # The toolset's cross-tool instructions and the tool spec are both present.
    assert "Always search first." in system_prompt
    assert "`search_documents`" in system_prompt
    # The wire spec is flattened to the toolset's member tool.
    assert [spec.name for spec in llm.tools[0]] == ["search_documents"]


def test_rag_toolset_provides_the_rag_tools_and_workflow():
    toolset = RAG_TOOLSET

    assert toolset.name == "RAG"
    assert [tool.name for tool in toolset.tools] == [
        "search_documents",
        "inspect_table",
        "inspect_table_relationships",
        "sql_query_table",
        "sql_query_documents",
    ]
    # The cross-tool orchestration lives in the toolset, not the tools.
    assert "search_documents" in toolset.instructions
    assert "sql_query_table" in toolset.instructions


def test_duplicate_tool_names_across_a_toolset_and_a_bare_tool_are_deduped():
    toolset = AgentToolset(name="Dup set", tools=[make_tool("dup")], instructions="x")
    agent = RagAgentService(
        rag_service=build_rag_service(FakeLLM(), NoopRetriever()),
        tools=[toolset, make_tool("dup")],
    )
    # The first occurrence wins; the duplicate is dropped, not an error.
    assert [tool.name for tool in agent._tools] == ["dup"]
    assert [spec.name for spec in agent._specs] == ["dup"]


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
        "#[s1:c1]": {"origin_source_id": "s1", "chunk_id": "c1"},
        "#[s2:c2]": {"origin_source_id": "s2", "chunk_id": "c2"},
    }
    # Plain brackets, ranks, and links are not citations (no leading #).
    assert extract_chunk_refs("[1] origin=s1 chunk=c1") == {}
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
                ToolCall(id="c1", name="inspect_table", arguments={"source_id": "tbl-src"}),
                ToolCall(id="c2", name="inspect_table_relationships", arguments={"source_id": "tbl-src"}),
            ),
        ),
        RawResult(content="Done."),
    )
    agent = make_agent(
        llm,
        [
            make_tool("inspect_table", "columns: region, amount"),
            make_tool("inspect_table_relationships", "no siblings"),
        ],
    )

    items = asyncio.run(
        drain(agent.ask_stream("describe the sales table", chat_id="s1"))
    )

    assert items[:5] == [
        StreamEvent(
            "tool_call", {"name": "inspect_table", "arguments": {"source_id": "tbl-src"}}
        ),
        StreamEvent(
            "tool_call",
            {
                "name": "inspect_table_relationships",
                "arguments": {"source_id": "tbl-src"},
            },
        ),
        StreamEvent(
            "tool_result",
            {"name": "inspect_table", "content": "columns: region, amount"},
        ),
        StreamEvent(
            "tool_result",
            {"name": "inspect_table_relationships", "content": "no siblings"},
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
