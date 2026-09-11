import asyncio

import pytest

from app.agent import Agent
from app.agent.events import (
    AgentEvent,
    AnswerDeltaEvent,
    AnswerEvent,
    ToolCallEvent,
    ToolResultEvent,
)
from app.agent.tools import AgentTools
from app.llm.base import RawResult, ToolCall

from fakes import FakeLLM, drain, make_tool, tool_call_response

FULL_ANSWER = "The garden plan is in the report."


def scripted_llm():
    return FakeLLM(
        tool_call_response("search_documents", {"query": "garden"}),
        RawResult(content=FULL_ANSWER),
    )


def scripted_agent(tools):
    return Agent(scripted_llm(), tools)


def test_ask_stream_yields_the_run_as_events():
    tools = AgentTools([make_tool("search_documents", "found: garden plan")])
    agent = scripted_agent(tools)

    events = asyncio.run(drain(agent.ask_stream("Where is the garden plan?")))

    # The answer streams as deltas, then the canonical answer terminates the
    # run.
    assert events == [
        ToolCallEvent(name="search_documents", arguments={"query": "garden"}),
        ToolResultEvent(name="search_documents", content="found: garden plan"),
        AnswerDeltaEvent(content=FULL_ANSWER),
        AnswerEvent(content=FULL_ANSWER),
    ]


def test_ask_stream_yields_the_answer_token_by_token():
    tools = AgentTools([make_tool("search_documents", "found: garden plan")])
    llm = FakeLLM(
        tool_call_response("search_documents", {"query": "garden"}),
        RawResult(content=FULL_ANSWER),
        chunk_size=4,
    )
    agent = Agent(llm, tools)

    events = asyncio.run(drain(agent.ask_stream("Where is the garden plan?")))

    deltas = [event for event in events if isinstance(event, AnswerDeltaEvent)]
    # The tokens arrive in order and compose the answer...
    assert "".join(delta.content for delta in deltas) == FULL_ANSWER
    assert len(deltas) == (len(FULL_ANSWER) + 3) // 4
    # ...and the last delta precedes the canonical answer, which terminates
    # the run.
    assert isinstance(events[-2], AnswerDeltaEvent)
    assert events[-1] == AnswerEvent(content=FULL_ANSWER)


def test_ask_is_the_last_event_of_the_stream():
    tools = AgentTools([make_tool("search_documents", "found: garden plan")])

    answer = asyncio.run(scripted_agent(tools).ask("Where is the garden plan?"))
    events = asyncio.run(drain(scripted_agent(tools).ask_stream("Where is the garden plan?")))

    final = events[-1]
    assert isinstance(final, AnswerEvent)
    assert final.content == answer
    assert all(isinstance(event, AgentEvent) for event in events)


def test_ask_stream_parallel_calls_yield_every_step_in_order():
    tools = AgentTools(
        [
            make_tool("list_tables", "one table: sales"),
            make_tool("inspect_table", "columns: region, amount"),
        ]
    )
    llm = FakeLLM(
        RawResult(
            content="",
            tool_calls=(
                ToolCall(id="c1", name="list_tables", arguments={}),
                ToolCall(id="c2", name="inspect_table", arguments={"table": "sales"}),
            ),
        ),
        RawResult(content="Done."),
    )
    agent = Agent(llm, tools)

    events = asyncio.run(drain(agent.ask_stream("describe the sales table")))

    assert events == [
        ToolCallEvent(name="list_tables", arguments={}),
        ToolCallEvent(name="inspect_table", arguments={"table": "sales"}),
        ToolResultEvent(name="list_tables", content="one table: sales"),
        ToolResultEvent(name="inspect_table", content="columns: region, amount"),
        AnswerDeltaEvent(content="Done."),
        AnswerEvent(content="Done."),
    ]


def test_ask_stream_streams_tool_turn_commentary_before_the_call():
    # A turn that carries both text and tool calls streams its text as
    # deltas, then reports the call: the commentary is visible, but it is
    # not the answer.
    tools = AgentTools([make_tool("search_documents", "found")])
    llm = FakeLLM(
        RawResult(
            content="checking now",
            tool_calls=(ToolCall(id="c1", name="search_documents", arguments={}),),
        ),
        RawResult(content="Done."),
        chunk_size=3,
    )
    agent = Agent(llm, tools)

    events = asyncio.run(drain(agent.ask_stream("q")))

    assert events[:4] == [
        AnswerDeltaEvent(content="che"),
        AnswerDeltaEvent(content="cki"),
        AnswerDeltaEvent(content="ng "),
        AnswerDeltaEvent(content="now"),
    ]
    assert events[4] == ToolCallEvent(name="search_documents", arguments={})
    assert events[5] == ToolResultEvent(name="search_documents", content="found")
    assert events[-1] == AnswerEvent(content="Done.")


def test_ask_stream_carries_tool_errors_as_results():
    tools = AgentTools([make_tool("search_documents")])
    llm = FakeLLM(
        tool_call_response("nope", {"x": 1}),
        RawResult(content="Recovered."),
    )
    agent = Agent(llm, tools)

    events = asyncio.run(drain(agent.ask_stream("q")))

    error = [
        event
        for event in events
        if isinstance(event, ToolResultEvent) and "unknown tool" in event.content
    ]
    assert len(error) == 1
    assert isinstance(events[-1], AnswerEvent)


def test_ask_raises_when_the_run_never_answers():
    tools = AgentTools([make_tool("search_documents", "still searching")])
    # The fake repeats its last scripted call, so the model keeps requesting
    # the tool even after the budget forces a tool-free final round.
    llm = FakeLLM(tool_call_response("search_documents"))
    agent = Agent(llm, tools, max_tool_rounds=1)

    with pytest.raises(RuntimeError):
        asyncio.run(agent.ask("loop?"))


def test_ask_stream_ends_silently_when_the_run_never_answers():
    tools = AgentTools([make_tool("search_documents", "still searching")])
    llm = FakeLLM(tool_call_response("search_documents"))
    agent = Agent(llm, tools, max_tool_rounds=1)

    events = asyncio.run(drain(agent.ask_stream("loop?")))

    assert all(isinstance(event, (ToolCallEvent, ToolResultEvent)) for event in events)
    assert not any(isinstance(event, AnswerEvent) for event in events)
