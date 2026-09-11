import asyncio

from app.agent import Agent
from app.agent.tools import AgentTools
from app.llm.base import RawResult, ToolCall

from fakes import FakeLLM, make_tool, tool_call_response


def test_agent_uses_a_tool_then_answers():
    calls = []
    tools = AgentTools(
        [make_tool("search_documents", "found: garden plan", calls)]
    )
    llm = FakeLLM(
        tool_call_response("search_documents", {"query": "garden"}),
        RawResult(content="The garden plan is in the report."),
    )
    agent = Agent(llm, tools)

    answer = asyncio.run(agent.ask("Where is the garden plan?"))

    assert answer == "The garden plan is in the report."
    assert calls == [("search_documents", {"query": "garden"})]
    # The first LLM call was made with the tool spec; the tool result was
    # fed back before the final call.
    assert [spec.name for spec in llm.tools[0]] == ["search_documents"]
    assert any(
        "found: garden plan" in str(message.content) for message in llm.calls[1]
    )


def test_agent_answers_without_tools_when_none_are_requested():
    calls = []
    tools = AgentTools([make_tool("search_documents", calls=calls)])
    llm = FakeLLM(RawResult(content="Direct answer."))
    agent = Agent(llm, tools)

    assert asyncio.run(agent.ask("hi")) == "Direct answer."
    assert calls == []


def test_agent_executes_several_tool_rounds():
    calls = []
    tools = AgentTools(
        [
            make_tool("list_tables", "one table: sales", calls),
            make_tool("query_table", "total is 35", calls),
        ]
    )
    llm = FakeLLM(
        tool_call_response("list_tables"),
        tool_call_response("query_table", {"table": "sales", "sql": "SELECT 1"}),
        RawResult(content="The total is 35."),
    )
    agent = Agent(llm, tools)

    answer = asyncio.run(agent.ask("total?"))

    assert answer == "The total is 35."
    assert [name for name, _ in calls] == ["list_tables", "query_table"]


def test_agent_runs_a_round_of_parallel_tool_calls_at_once():
    calls = []
    tools = AgentTools(
        [
            make_tool("list_tables", "one table: sales", calls),
            make_tool("inspect_table", "columns: region, amount", calls),
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

    answer = asyncio.run(agent.ask("describe the sales table"))

    assert answer == "Done."
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
    assert [spec.name for spec in llm.tools[1]] == [
        "list_tables",
        "inspect_table",
    ]


def test_agent_stops_at_the_tool_budget():
    calls = []
    tools = AgentTools([make_tool("search_documents", "still searching", calls)])
    llm = FakeLLM(
        tool_call_response("search_documents"),
        RawResult(content="Best effort answer."),
    )
    agent = Agent(llm, tools, max_tool_rounds=1)

    answer = asyncio.run(agent.ask("loop?"))

    assert answer == "Best effort answer."
    # The first round used the tool; the budgeted-out call went to the LLM
    # without tools and produced the final answer.
    assert len(calls) == 1
    assert llm.tools[-1] == []


def test_agent_conversation_sends_history_to_the_model():
    tools = AgentTools([make_tool("search_documents", "evidence")])
    llm = FakeLLM(
        RawResult(content="First answer."),
        RawResult(content="Second answer."),
    )
    agent = Agent(llm, tools)

    first = asyncio.run(agent.ask("Tell me about sales."))
    second = asyncio.run(
        agent.ask(
            "How about the south region?",
            history=[("user", "Tell me about sales."), ("assistant", first)],
        )
    )

    assert first == "First answer."
    assert second == "Second answer."
    # The second turn's LLM call carries the prior exchange, oldest first,
    # between the system prompt and the current question.
    conversation = [
        (message.role, message.content)
        for message in llm.calls[-1]
        if message.role != "system"
    ]
    assert conversation == [
        ("user", "Tell me about sales."),
        ("assistant", "First answer."),
        ("user", "How about the south region?"),
    ]


def test_tool_errors_are_seen_by_the_model_not_raised():
    tools = AgentTools([make_tool("search_documents")])
    llm = FakeLLM(
        tool_call_response("nope", {"x": 1}),
        RawResult(content="Recovered."),
    )
    agent = Agent(llm, tools)

    assert asyncio.run(agent.ask("q")) == "Recovered."
    # The unknown tool call came back as an error string for the model.
    feedback = [m for m in llm.calls[1] if "unknown tool" in str(m.content)]
    assert len(feedback) == 1
