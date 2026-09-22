from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass

from app.agent_tools.evidence import EvidenceIndex
from app.services.rag import RagService

# The async function the agent calls for a tool: parsed arguments in,
# the text the model reads back out.
ToolExecutor = Callable[[dict], Awaitable[str]]


@dataclass
class RunContext:
    """Per-answer state handed to the tools when their executors are built.

    ``chat_id`` scopes lookups to the chat; ``query`` is the user's question
    for this run; ``evidence`` is the index the chunk-surfacing tools register
    into, so the answer can cite a chunk by a small integer the service later
    resolves to the real ids."""

    chat_id: str | None
    query: str
    evidence: EvidenceIndex


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
    def create_executor(self, rag_service: RagService, context: RunContext) -> ToolExecutor:
        """Return the async execute function the agent uses for this tool.

        Called once per answer with the RAG service and the per-answer context
        (the chat scope plus the evidence index). The returned closure may
        capture whatever it needs; chunk-surfacing tools register their
        results in the context's evidence index so the answer can cite them by
        number.
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


class AgentToolset:
    """A group of tools plus the cross-tool instructions for orchestrating
    them.

    The instructions live here (not in the individual tools), so each tool
    stays decoupled and describes only itself. A toolset is transparent to the
    tool-call wire format: its member tools are what the model actually calls,
    and its rendered block (name + instructions + specs) is what the system
    prompt shows.
    """

    def __init__(
        self, name: str, tools: Sequence[AgentTool], instructions: str
    ) -> None:
        self.name = name
        self._tools = list(tools)
        self.instructions = instructions
        seen = set()
        for tool in self._tools:
            if tool.name in seen:
                raise ValueError(
                    f"Tool name '{tool.name}' appears more than once in the toolset."
                )
            seen.add(tool.name)

    @property
    def tools(self) -> list[AgentTool]:
        return self._tools

    def outline(self) -> str:
        """The spec of each tool in the set: name, description, parameters."""
        return tools_outline(self._tools)

    def render(self) -> str:
        """The set's prompt block: its name, the cross-tool instructions, and
        each tool's spec."""
        return f"Toolset name: {self.name}\n\nToolset Description:\n{self.instructions}\n\nTools:\n{self.outline()}"


def flatten_tools(items: Sequence[AgentTool | AgentToolset]) -> list[AgentTool]:
    """All the tools in a mix of bare tools and toolsets, in order. Toolsets
    contribute their member tools; the toolset wrappers themselves are not
    tools the model calls."""
    tools: list[AgentTool] = []
    for item in items:
        if isinstance(item, AgentToolset):
            tools.extend(item.tools)
        else:
            tools.append(item)
    return tools


def render_tool_blocks(items: Sequence[AgentTool | AgentToolset]) -> str:
    """The system-prompt tools section, rendered from a mix of bare tools and
    toolsets. A toolset contributes its name, instructions, and its tools'
    specs; bare tools are grouped under a single Tools: header."""
    blocks: list[str] = []
    bare: list[AgentTool] = []
    for item in items:
        if isinstance(item, AgentToolset):
            blocks.append(item.render())
        else:
            bare.append(item)
    if bare:
        blocks.append("Tools:\n" + tools_outline(bare))
    return "\n\n".join(blocks)
