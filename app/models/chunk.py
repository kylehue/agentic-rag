from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4


@dataclass
class IngestedChunk:
    """A chunk produced by a plugin during ingestion."""

    plugin: str
    chunk_id: str = field(default_factory=lambda: str(uuid4()))

    # --- file info (used when saving a chunk as file) ---
    file_filename: str | None = None
    file_content_type: str | None = None
    file_bytes: bytes | None = None

    # --- chunk info ---
    text: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievedChunk:
    """The retrieved document chunk during retrieval process."""

    chunk_id: str
    source_id: str
    plugin: str
    text: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        **overrides: Any,
    ) -> "RetrievedChunk":
        values = {
            "chunk_id": data["chunk_id"],
            "source_id": data["source_id"],
            "plugin": data["plugin"],
            "text": data["text"],
            "metadata": data.get("metadata") or {},
            "score": data.get("score", 0.0),
        }

        values.update(overrides)

        return cls(**values)
