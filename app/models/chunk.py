from dataclasses import dataclass, field
from enum import Enum
from typing import Any
from uuid import uuid4

from unstructured.documents.elements import Element


class ChunkCategory(str, Enum):
    """The supported source types used to choose processing and retrieval behavior."""

    DOCUMENT = "document"
    SPREADSHEET = "spreadsheet"
    IMAGE = "image"


@dataclass
class IngestedChunk:
    """The persistent record describing a document chunk."""

    id: str = field(default_factory=lambda: str(uuid4()))

    # --- file info (used when saving a chunk as file) ---
    file_filename: str | None = None
    file_content_type: str | None = None
    file_bytes: bytes | None = None

    # --- chunk info ---
    category: ChunkCategory = ChunkCategory.DOCUMENT
    text: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    orig_elements: list[Element] | None = None  # original Unstructured.io elements


@dataclass
class RetrievedChunk:
    """The retrieved document chunk during retrieval process."""

    chunk_id: str
    source_id: str
    category: ChunkCategory
    text: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.category, ChunkCategory):
            self.category = ChunkCategory(self.category)
