from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ToolInfo:
    """One tool an MCP server offers, captured generically.

    The fields mirror what the MCP `list_tools` result carries, so an adapter
    can turn them straight into an agent tool without knowing about MCP.
    """

    name: str
    description: str
    input_schema: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ToolsetInfo:
    """A connected MCP server's tools, grouped by the server's name."""

    name: str
    tools: list[ToolInfo] = field(default_factory=list)
