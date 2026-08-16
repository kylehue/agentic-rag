from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from pydantic import BaseModel
from unstructured.documents.elements import Element


class DocumentCategory(str, Enum):
    DOCUMENT = "document"
    SPREADSHEET = "spreadsheet"
    IMAGE = "image"


class Document(BaseModel):
    id: str
    filename: str
    extension: str
    path: str
    category: DocumentCategory


@dataclass
class DocumentChunk:
    id: str
    text: str
    document: Document
    metadata: dict[str, Any]
    # chunk.text is only used for vector search but the image itself should be sent to llm
    binary_content: bytes | None = None
    binary_mime_type: str | None = None


@dataclass
class DocumentProcessorChunk(DocumentChunk):
    """A chunk while processing, retaining its source elements from Unstructured.io"""

    orig_elements: list[Element] = field(default_factory=list)
