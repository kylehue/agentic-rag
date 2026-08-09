from app.models.document import Document, DocumentCategory, DocumentChunk

from app.processors.table import process_table
from app.processors.text import process_text
from app.processors.image import process_image
from unstructured.documents.elements import Element, Table, Image


async def process(
    document: Document,
    elements: list[Element],
) -> list[DocumentChunk]:
    tables = []
    images = []
    for i, element in enumerate(elements):
        if isinstance(element, Table):
            tables.append(element)
        if isinstance(element, Image):
            images.append(element)

    chunks = []

    # text documents
    if document.category is DocumentCategory.DOCUMENT:
        text_chunks = await process_text(document, elements)
        chunks.extend(text_chunks)

        # pass every chunk that has image to image processor along with additional context
        for i, chunk in enumerate(text_chunks):
            pass

    # spreadsheet documents
    if document.category is DocumentCategory.SPREADSHEET:
        chunks.extend(await process_table(document, elements))

    # image documents
    if document.category is DocumentCategory.IMAGE:
        chunks.extend(await process_image(document, elements))

    return chunks
