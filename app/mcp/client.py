import logging
from collections.abc import Sequence
from typing import Protocol

from app.mcp.models import ToolInfo, ToolsetInfo

logger = logging.getLogger(__name__)


class ManagedServer(Protocol):
    """The surface McpClient uses from each server. It is a protocol (not the
    concrete McpServer) so the client stays decoupled from one implementation
    and is testable with a stand-in."""

    @property
    def name(self) -> str: ...
    @property
    def connected(self) -> bool: ...
    async def connect(self) -> None: ...
    async def close(self) -> None: ...
    async def list_tools(self) -> list[ToolInfo]: ...
    async def call_tool(self, tool_name: str, arguments: dict) -> str: ...


class McpClient:
    """Manages a set of MCP server connections and their discovered tools.

    `connect` opens each server and lists its tools, caching the result. A
    server that fails to connect or list tools is skipped with a warning, so
    the app still starts (its tools are just absent). `call_tool` routes a
    call to the owning server.
    """

    def __init__(self, servers: Sequence[ManagedServer]) -> None:
        self._servers = list(servers)
        self._toolsets: list[ToolsetInfo] = []

    async def connect(self) -> None:
        for server in self._servers:
            try:
                await server.connect()
                tools = await server.list_tools()
            except Exception as exc:
                logger.warning("MCP server %r unavailable: %s", server.name, exc)
                await server.close()
                continue
            self._toolsets.append(ToolsetInfo(name=server.name, tools=tools))

    async def close(self) -> None:
        for server in self._servers:
            await server.close()

    @property
    def toolsets(self) -> list[ToolsetInfo]:
        return self._toolsets

    async def call_tool(self, server_name: str, tool_name: str, arguments: dict) -> str:
        for server in self._servers:
            if server.name == server_name and server.connected:
                return await server.call_tool(tool_name, arguments)
        raise ValueError(f"MCP server '{server_name}' is not connected.")
