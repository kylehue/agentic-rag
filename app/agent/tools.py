from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass

from app.llm.base import ToolSpec

ToolExecutor = Callable[[dict], Awaitable[str]]


@dataclass(frozen=True)
class AgentTool:
    """One tool the agent can call.

    `execute` receives the model's parsed arguments and returns the text the
    model reads back. Tool failures are returned as error strings, not
    raised: the agent should see the error and be able to adapt.
    """

    name: str
    description: str
    parameters: dict
    execute: ToolExecutor


class AgentTools:
    """The set of tools offered to the agent.

    The orchestration graph and the LLM tool specs both derive from this
    list; adding a tool is adding an `AgentTool` to the set.
    """

    def __init__(self, tools: Sequence[AgentTool]) -> None:
        self._by_name: dict[str, AgentTool] = {}
        for tool in tools:
            if tool.name in self._by_name:
                raise ValueError(f"Tool name '{tool.name}' is already registered.")
            self._by_name[tool.name] = tool

    def specs(self) -> list[ToolSpec]:
        """The tools in the LLM wire format, in registration order."""
        return [
            ToolSpec(
                name=tool.name,
                description=tool.description,
                parameters=tool.parameters,
            )
            for tool in self._by_name.values()
        ]

    async def execute(self, name: str, arguments: dict) -> str:
        tool = self._by_name.get(name)
        if tool is None:
            return f"Error: unknown tool '{name}'."
        try:
            return await tool.execute(arguments)
        except Exception as exc:
            return f"Error in tool '{name}': {exc}"

    def outline(self) -> str:
        """The tools as a system-prompt outline: one entry per tool with its
        description and a compact parameter summary.

        Derived from the tool definitions, so a prompt built on it can never
        drift from the tools the agent actually gets. (The full JSON schemas
        with per-parameter descriptions still go to the model via `specs`;
        the outline is the strategy-level summary.)
        """
        lines = []
        for tool in self._by_name.values():
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
