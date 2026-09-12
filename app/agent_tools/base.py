from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable, Sequence

from app.services.rag import RagService

# The async function the agent calls for a tool: parsed arguments in,
# the text the model reads back out.
ToolExecutor = Callable[[dict], Awaitable[str]]


class AgentTool(ABC):
    """Base class for the agent's tools (plugin-like: self-describing, wired
    to the RAG service when its executor is created).

    Subclasses declare `name`, `description`, and `parameters` (the JSON
    schema the model sees), and implement `create_executor`, which returns
    the async execute function the agent uses for this tool.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """The tool name the model calls it by."""

    @property
    @abstractmethod
    def description(self) -> str:
        """What the tool does, for the model and the system prompt outline."""

    @property
    @abstractmethod
    def parameters(self) -> dict:
        """The tool's JSON schema parameters."""

    @abstractmethod
    def create_executor(self, rag_service: RagService) -> ToolExecutor:
        """Return the async execute function the agent uses for this tool.

        Called once per answer with the RAG service; the returned closure
        may capture whatever it needs from the service.
        """


def tools_outline(tools: Sequence[AgentTool]) -> str:
    """The tools as a system-prompt outline: one entry per tool with its
    description and a compact parameter summary.

    Derived from the tool definitions, so a prompt built on it can never
    drift from the tools the agent actually gets. (The full JSON schemas
    with per-parameter descriptions still go to the model on every call;
    the outline is the strategy-level summary.)
    """
    lines = []
    for tool in tools:
        lines.append(f"- `{tool.name}`: {tool.description}")
        summary = _parameters_summary(tool.parameters)
        if summary:
            lines.append(f"  Parameters: {summary}")
    return "\n".join(lines)


def _parameters_summary(parameters: dict) -> str:
    """A compact one-line summary of a JSON schema's parameters:
    `name (type, required|optional)` per property, in declaration order."""
    properties = parameters.get("properties") or {}
    required = set(parameters.get("required") or [])
    return ", ".join(
        f"{name} ({prop.get('type', 'any')}, "
        f"{'required' if name in required else 'optional'})"
        for name, prop in properties.items()
    )
