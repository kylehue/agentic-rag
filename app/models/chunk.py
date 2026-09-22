from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Self
from uuid import uuid4

from app.models.content import ImageContent


@dataclass
class IngestedChunk:
    """Base for a chunk a plugin produces during ingestion.

    Carries identity and metadata only; the concrete subclasses carry the
    payload (text or image). `key` is an optional dedup identity: chunks that
    share a non-null `key` are the same logical unit, so retrieval keeps only
    the best of them. Null (the default) means the chunk is its own unit.
    """

    plugin: str
    chunk_id: str = field(default_factory=lambda: str(uuid4()))
    metadata: dict[str, Any] = field(default_factory=dict)
    key: str | None = None


@dataclass
class IngestedTextChunk(IngestedChunk):
    """A text chunk: searchable text content, embedded as text."""

    text: str = ""


@dataclass
class IngestedImageChunk(IngestedChunk):
    """An image chunk: the image bytes, embedded as an image.

    The image is also persisted as a file (addressable by source_id) for
    on-demand viewing; the bytes here are what the image embedder consumes.
    """

    image: ImageContent | None = None

    def __post_init__(self):
        if self.image is None:
            raise ValueError("IngestedImageChunk requires an image.")


@dataclass
class RetrievedChunk:
    """Base for a retrieved chunk: identity, lineage, and score.

    Concrete subclasses carry their payload (text). A retrieved image chunk
    carries no image bytes; the image is loaded on demand from file storage
    via its source_id (loading at retrieval would waste resources).
    """

    chunk_id: str
    source_id: str
    origin_source_id: str
    plugin: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)
    parent_source_id: str | None = None
    chat_id: str | None = None
    # Optional dedup identity (see IngestedChunk.key); null means no dedup.
    key: str | None = None

    @classmethod
    def from_dict(cls, data: dict, **overrides: Any) -> Self:
        values = {
            "chunk_id": data["chunk_id"],
            "source_id": data["source_id"],
            "origin_source_id": data["origin_source_id"],
            "plugin": data["plugin"],
            "metadata": data.get("metadata") or {},
            "score": data.get("score", 0.0),
            "parent_source_id": data.get("parent_source_id"),
            "chat_id": data.get("chat_id"),
            "key": data.get("key"),
        }

        values.update(overrides)

        return cls(**values)


@dataclass
class RetrievedTextChunk(RetrievedChunk):
    """A retrieved text chunk: its searchable text."""

    text: str = ""


@dataclass
class RetrievedImageChunk(RetrievedChunk):
    """A retrieved image chunk (the image is loaded on demand via source_id)."""
