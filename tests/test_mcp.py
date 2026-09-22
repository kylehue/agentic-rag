import asyncio

from app.agent_tools import RAG_TOOLSET, EvidenceIndex, RunContext
from app.agent_tools.mcp import build_mcp_toolsets
from app.mcp import McpClient
from app.mcp.models import ToolInfo
from app.retrievers.base import Retriever
from app.services.rag_agent import RagAgentService

from fakes import FakeLLM, build_rag_service


class NoopRetriever(Retriever):
    async def retrieve(self, user_query, where=None):
        return []


class FakeMcpServer:
    """Stands in for McpServer: the McpClient only uses name, connected,
    connect, close, list_tools, and call_tool."""

    def __init__(
        self,
        name,
        tools=None,
        *,
        connect_error=None,
        list_error=None,
        result="ok",
    ):
        self._name = name
        self._tools = tools or []
        self._connect_error = connect_error
        self._list_error = list_error
        self._result = result
        self.connected = False
        self.calls = []

    @property
    def name(self):
        return self._name

    async def connect(self):
        if self._connect_error:
            raise self._connect_error
        self.connected = True

    async def close(self):
        self.connected = False

    async def list_tools(self):
        if self._list_error:
            raise self._list_error
        return self._tools

    async def call_tool(self, tool_name, arguments):
        self.calls.append((tool_name, arguments))
        return self._result


def make_tool_info(name, description="desc", schema=None):
    return ToolInfo(
        name=name,
        description=description,
        input_schema=schema or {"type": "object", "properties": {}},
    )


def test_mcp_client_skips_unavailable_servers():
    good = FakeMcpServer("good", tools=[make_tool_info("web_search")])
    bad = FakeMcpServer("bad", connect_error=RuntimeError("no network"))
    client = McpClient([good, bad])

    asyncio.run(client.connect())

    # The good server's tools are cached; the bad one is skipped.
    assert [toolset.name for toolset in client.toolsets] == ["good"]
    assert [tool.name for tool in client.toolsets[0].tools] == ["web_search"]
    assert good.connected is True
    assert bad.connected is False
    asyncio.run(client.close())


def test_mcp_client_closes_a_server_that_fails_to_list_tools():
    good = FakeMcpServer("good", tools=[make_tool_info("web_search")])
    bad = FakeMcpServer("bad", tools=[make_tool_info("x")], list_error=RuntimeError("boom"))
    client = McpClient([good, bad])

    asyncio.run(client.connect())

    assert [toolset.name for toolset in client.toolsets] == ["good"]
    # The failing server was connected, then closed after the list failure.
    assert bad.connected is False


def test_mcp_client_routes_calls_to_the_owning_server():
    good = FakeMcpServer("good", tools=[make_tool_info("web_search")])
    other = FakeMcpServer("other", tools=[make_tool_info("lookup")])
    client = McpClient([good, other])
    asyncio.run(client.connect())

    result = asyncio.run(client.call_tool("other", "lookup", {"q": "x"}))

    assert result == "ok"
    assert other.calls == [("lookup", {"q": "x"})]
    assert good.calls == []
    asyncio.run(client.close())


def test_build_mcp_toolsets_groups_tools_by_server_and_namespaces_them():
    server = FakeMcpServer(
        "parallel",
        tools=[
            make_tool_info(
                "web_search",
                "Search the web.",
                {"type": "object", "properties": {"query": {"type": "string"}}},
            )
        ],
    )
    client = McpClient([server])
    asyncio.run(client.connect())

    toolsets = build_mcp_toolsets(client)
    assert len(toolsets) == 1
    toolset = toolsets[0]
    assert toolset.name == "parallel"
    # The tool is namespaced so it cannot shadow a built-in tool.
    assert [tool.name for tool in toolset.tools] == ["parallel__web_search"]
    tool = toolset.tools[0]
    assert tool.description == "Search the web."
    assert tool.parameters["properties"]["query"]["type"] == "string"

    # The executor routes the call through the client to the server, ignoring
    # the RAG service and chat it is given.
    executor = tool.create_executor(
        rag_service=build_rag_service(FakeLLM("ok"), NoopRetriever()),
            context=RunContext(chat_id=None, query="q", evidence=EvidenceIndex()),
    )

    async def run() -> str:
        return await executor({"query": "hello"})

    assert asyncio.run(run()) == "ok"
    assert server.calls == [("web_search", {"query": "hello"})]
    asyncio.run(client.close())


def test_agent_merges_mcp_tools_on_initialize():
    server = FakeMcpServer("parallel", tools=[make_tool_info("web_search")])
    client = McpClient([server])
    asyncio.run(client.connect())

    agent = RagAgentService(
        rag_service=build_rag_service(FakeLLM("ok"), NoopRetriever()),
        tools=[RAG_TOOLSET],
        mcp_client=client,
    )

    # Before initialize, only the static RAG tools are present.
    assert "parallel__web_search" not in [tool.name for tool in agent._tools]

    asyncio.run(agent.initialize())

    # After initialize, the MCP tool is merged in (namespaced), and the
    # prompt carries the MCP toolset.
    names = [tool.name for tool in agent._tools]
    assert "parallel__web_search" in names
    assert "search_documents" in names
    assert "parallel" in agent._system_prompt
    asyncio.run(client.close())
