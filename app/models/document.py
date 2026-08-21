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
class Document:
    """The persistent record describing a document chunk."""

    # --- file info ---
    file_filename: str
    file_content_type: str
    file_bytes: bytes | None = None

    # --- chunk info ---
    category: DocumentCategory = DocumentCategory.DOCUMENT
    text: str = ""
    metadata: dict[str, Any] = {}
    # chunk's original Unstructured.io elements, used for processing
    orig_elements: list[Element] | None = None
    id: str = str(uuid4())
