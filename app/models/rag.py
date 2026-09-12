from dataclasses import dataclass, field


@dataclass(frozen=True)
class RagAnswer:
    query: str
    answer: str
    chunk_refs: dict[str, dict[str, str]] = field(default_factory=dict)
