import logging
from contextlib import AsyncExitStack
from typing import Any

from app.mcp.models import ToolInfo

logger = logging.getLogger(__name__)


class McpServer:
    """One connected MCP server endpoint.

    Opens a transport connection (streamable HTTP by default, or SSE) plus an
    MCP session, then lists the server's tools and routes tool calls to it.
    The MCP SDK imports live inside `connect` so this module (and the app) can
    load even when `mcp` is not installed and MCP is disabled.
    """

    def __init__(
        self,
        name: str,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        transport: str = "streamable_http",
    ) -> None:
        self._name = name
        self._url = url
        self._headers = headers or {}
        self._transport = transport
        self._stack: AsyncExitStack | None = None
        # The open MCP session, an opaque handle to the connected server.
        self._session: Any = None

    @property
    def name(self) -> str:
        return self._name

    @property
    def connected(self) -> bool:
        return self._session is not None

    async def connect(self) -> None:
        import httpx
        from mcp import ClientSession
        from mcp.client.sse import sse_client
        from mcp.client.streamable_http import streamable_http_client

        # One exit stack manages the transport, the http client, and the
        # session; close() releases them as a unit.
        self._stack = AsyncExitStack()
        if self._transport == "streamable_http":
            http = await self._stack.enter_async_context(
                httpx.AsyncClient(headers=self._headers)
            )
            read, write = await self._stack.enter_async_context(
                # The MCP SDK aliases httpx as httpx2, so its stub types the
                # client as httpx2.AsyncClient; at runtime it is plain httpx.
                streamable_http_client(self._url, http_client=http)  # type: ignore[arg-type]
            )
        elif self._transport == "sse":
            read, write = await self._stack.enter_async_context(
                sse_client(self._url, headers=self._headers or None)
            )
        else:
            await self._stack.aclose()
            self._stack = None
            raise ValueError(f"Unsupported MCP transport '{self._transport}'.")

        self._session = await self._stack.enter_async_context(ClientSession(read, write))
        await self._session.initialize()

    async def close(self) -> None:
        if self._stack is not None:
            await self._stack.aclose()
        self._stack = None
        self._session = None

    async def list_tools(self) -> list[ToolInfo]:
        assert self._session is not None, "MCP server is not connected."
        result = await self._session.list_tools()
        return [
            ToolInfo(
                name=tool.name,
                description=tool.description or "",
                input_schema=tool.input_schema or {"type": "object", "properties": {}},
            )
            for tool in result.tools
        ]

    async def call_tool(self, tool_name: str, arguments: dict) -> str:
        assert self._session is not None, "MCP server is not connected."
        result = await self._session.call_tool(tool_name, arguments)
        if getattr(result, "isError", False):
            text = self._text_of(result)
            return f"Error from the MCP tool: {text}" if text else "Error from the MCP tool."
        return self._text_of(result) or "(the MCP tool returned no text)"

    @staticmethod
    def _text_of(result) -> str:
        # A tool result's content is a list of blocks; keep the text ones and
        # ignore any the model cannot read (images, resources).
        parts = []
        for block in getattr(result, "content", None) or []:
            if getattr(block, "type", None) == "text":
                parts.append(block.text)
        return "\n".join(parts)
