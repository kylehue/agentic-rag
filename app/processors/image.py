from app.models.document import Document, DocumentChunk
from unstructured.documents.elements import Element


async def process_image(
    document: Document,
    elements: list[Element],
) -> list[DocumentChunk]:
    return []
