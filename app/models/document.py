from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict
from unstructured.documents.elements import Element


class DocumentCategory(str, Enum):
    """The supported source types used to choose processing and retrieval behavior."""

    DOCUMENT = "document"
    SPREADSHEET = "spreadsheet"
    IMAGE = "image"


class Document(BaseModel):
    """The persistent record describing one uploaded source file."""

    model_config = ConfigDict(arbitrary_types_allowed=True)  # allow orig_elements

    # --- file info ---
    id: str
    filename: str
    extension: str
    path: str
    category: DocumentCategory

    # --- chunk info ---
    text: str = ""
    metadata: dict[str, Any] = {}
    # chunk.text is only used for vector search but the image itself should be sent to llm
    binary_content: bytes | None = None
    binary_mime_type: str | None = None
    # chunk's original Unstructured.io elements, used for processing
    orig_elements: list[Element] | None = None
