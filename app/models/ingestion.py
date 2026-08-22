from dataclasses import dataclass

from unstructured.documents.elements import Element

from app.llm.base import LLMProvider
from app.models.chunk import ChunkCategory


@dataclass
class ProcessorPayload:
    source_id: str
    source_filename: str
    source_bytes: bytes
    source_content_type: str
    category: ChunkCategory
    llm: LLMProvider
    elements: list[Element]
    is_embedded: bool = False
