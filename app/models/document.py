from enum import Enum

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


class DocumentChunk(BaseModel):
    id: str
    text: str
    document: Document
    orig_elements: list[Element]
    metadata: dict
