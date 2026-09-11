from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class StreamEvent:
    """One neutral item in an answer stream: a wire event name plus a payload.

    The service yields these (translated from the agent's domain events); the
    API serializes them without knowing about the agent. `payload` is a plain
    dict for step events (`tool_call`, `tool_result`) and the final `RagAnswer`
    for the terminal `answer` event, which the API schema-serializes.
    """

    name: str
    payload: Any = None
