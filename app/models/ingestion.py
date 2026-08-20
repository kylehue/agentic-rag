from dataclasses import dataclass

from unstructured.documents.elements import Element

from app.llm.base import LLMProvider
from app.models.document import DocumentCategory


@dataclass
class ProcessorPayload:
    file_id: str
    file_filename: str
    file_bytes: bytes
    file_content_type: str
    document_category: DocumentCategory
    llm: LLMProvider
    elements: list[Element]
