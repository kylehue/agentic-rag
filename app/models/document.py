from dataclasses import dataclass, field
from enum import Enum
from typing import Any
from uuid import uuid4

from unstructured.documents.elements import Element


class DocumentCategory(str, Enum):
    """The supported source types used to choose processing and retrieval behavior."""

    DOCUMENT = "document"
    SPREADSHEET = "spreadsheet"
    IMAGE = "image"


@dataclass
class DocumentChunk:
    """The persistent record describing a document chunk."""

    id: str = str(uuid4())

    # --- file info (used when saving a chunk as file) ---
    file_filename: str | None = None
    file_content_type: str | None = None
    file_bytes: bytes | None = None

    # --- chunk info ---
    category: DocumentCategory = DocumentCategory.DOCUMENT
    text: str = ""
    metadata: dict[str, Any] = {}
    orig_elements: list[Element] | None = None  # original Unstructured.io elements


@dataclass
class RetrievedDocumentChunk:
    """The retrieved document chunk during retrieval process."""

    chunk_id: str
    source_id: str
    text: str
    metadata: list[dict[str, Any]]
    score: float
