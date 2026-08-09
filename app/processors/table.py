from app.models.document import Document, DocumentChunk
from unstructured.documents.elements import Element


async def process_table(
    document: Document,
    elements: list[Element],
) -> list[DocumentChunk]:
    return []
