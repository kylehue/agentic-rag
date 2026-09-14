from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4


@dataclass
class IngestedChunk:
    """A chunk produced by a plugin during ingestion.

    A chunk is text plus metadata only. Plugins that need a file persisted for
    later use emit it with ``runtime.emit_file`` during processing; the chunk
    itself never carries bytes.
    """

    plugin: str
    chunk_id: str = field(default_factory=lambda: str(uuid4()))

    text: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievedChunk:
    """The retrieved document chunk during retrieval process.

    ``origin_source_id`` is always set (a top-level file is its own origin);
    only ``parent_source_id`` is optional. ``chat_id`` is the chat the chunk
    was ingested into (None for data predating chats).
    """

    chunk_id: str
    source_id: str
    origin_source_id: str
    plugin: str
    text: str
    score: float
    parent_source_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    chat_id: str | None = None

    @classmethod
    def from_dict(
        cls,
        data: dict,
        **overrides: Any,
    ) -> "RetrievedChunk":
        values = {
            "chunk_id": data["chunk_id"],
            "source_id": data["source_id"],
            "origin_source_id": data["origin_source_id"],
            "plugin": data["plugin"],
            "text": data["text"],
            "metadata": data.get("metadata") or {},
            "score": data.get("score", 0.0),
            "parent_source_id": data.get("parent_source_id"),
            "chat_id": data.get("chat_id"),
        }

        values.update(overrides)

        return cls(**values)
