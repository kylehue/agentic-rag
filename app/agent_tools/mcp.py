from app.agent_tools.base import AgentTool, AgentToolset
from app.mcp import McpClient
from app.mcp.models import ToolInfo

# The instructions a per-server MCP toolset shows in the prompt.
_MCP_TOOLSET_INSTRUCTIONS = "Tools provided by the {server} MCP server."


class McpTool(AgentTool):
    """One MCP server tool, adapted to the agent's tool interface.

    The model calls it by its namespaced name (server__tool); the executor
    routes the call through the MCP client to the owning server. The tool
    knows nothing about RAG, so it ignores the service and chat it is given.
    """

    def __init__(self, client: McpClient, server_name: str, info: ToolInfo) -> None:
        self._client = client
        self._server_name = server_name
        self._info = info

    @property
    def name(self) -> str:
        # Namespaced so an MCP tool can never shadow a built-in (or another
        # server's tool of the same name).
        return f"{self._server_name}__{self._info.name}"

    @property
    def description(self) -> str:
        return self._info.description or self._info.name

    @property
    def parameters(self) -> dict:
        return self._info.input_schema or {"type": "object", "properties": {}}

    def create_executor(self, rag_service, chat_id=None):
        client = self._client
        server_name = self._server_name
        tool_name = self._info.name

        async def execute(arguments: dict) -> str:
            return await client.call_tool(server_name, tool_name, arguments)

        return execute


def build_mcp_toolsets(client: McpClient) -> list[AgentToolset]:
    """One AgentToolset per connected MCP server, so the prompt shows each
    server's tools under their own name. Servers with no tools are skipped."""
    toolsets = []
    for info in client.toolsets:
        if not info.tools:
            continue
        toolsets.append(
            AgentToolset(
                name=info.name,
                tools=[McpTool(client, info.name, tool) for tool in info.tools],
                instructions=_MCP_TOOLSET_INSTRUCTIONS.format(server=info.name),
            )
        )
    return toolsets
