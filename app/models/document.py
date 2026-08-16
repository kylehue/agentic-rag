from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from pydantic import BaseModel
from unstructured.documents.elements import Element


class DocumentCategory(str, Enum):
    """The supported source types used to choose processing and retrieval behavior."""

    DOCUMENT = "document"
    SPREADSHEET = "spreadsheet"
    IMAGE = "image"


class Document(BaseModel):
    """The persistent record describing one uploaded source file."""

    id: str
    filename: str
    extension: str
    path: str
    category: DocumentCategory


@dataclass
class DocumentChunk:
    """One searchable piece of a document."""

    id: str
    text: str
    document: Document
    metadata: dict[str, Any]
    # chunk.text is only used for vector search but the image itself should be sent to llm
    binary_content: bytes | None = None
    binary_mime_type: str | None = None


@dataclass
class DocumentProcessorChunk(DocumentChunk):
    """A chunk during processing, retaining its source elements from Unstructured.io"""

    orig_elements: list[Element] = field(default_factory=list)
