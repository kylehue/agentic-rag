from dataclasses import dataclass
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
    orig_elements: list[Element]
    metadata: dict[str, Any]
