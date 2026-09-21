from dataclasses import dataclass, field


@dataclass(frozen=True)
class AgentEvent:
    """One observable step of an agent run."""


@dataclass(frozen=True)
class AnswerDeltaEvent(AgentEvent):
    """An increment of the model's text as the current turn streams.

    An advisory preview of the in-flight turn: if the turn ends with tool
    calls, its deltas were commentary (the trace carries only the tool
    call); if the turn is the answer, they compose it. The final
    `AnswerEvent` remains the canonical text.
    """

    content: str


@dataclass(frozen=True)
class ToolCallEvent(AgentEvent):
    """The model asked to call a tool (one per call, in the model's order)."""

    name: str
    arguments: dict


@dataclass(frozen=True)
class ToolResultEvent(AgentEvent):
    """A tool ran; `content` is what the model reads back (result or error)."""

    name: str
    content: str


@dataclass(frozen=True)
class AnswerEvent(AgentEvent):
    """The agent's final answer. The last event of a run.

    `chunk_refs` is the resolved citation map (``"#[n]"`` -> the real chunk
    ids), computed against the run's evidence index."""

    content: str
    chunk_refs: dict[str, dict[str, str]] = field(default_factory=dict)
