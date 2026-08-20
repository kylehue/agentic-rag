from app.models.document import Document, DocumentCategory
from app.llm.base import LLMProvider

from app.processors.table import process_table
from app.processors.text import process_text
from app.processors.image import process_image
from unstructured.documents.elements import Element


async def process(
    document: Document,
    elements: list[Element],
    llm: LLMProvider,
) -> list[Document]:
    """Process document and return every searchable chunk."""
    chunks = []

    # text documents
    if document.category is DocumentCategory.DOCUMENT:
        chunks.extend(await process_text(document, elements, llm))

    # spreadsheet documents
    elif document.category is DocumentCategory.SPREADSHEET:
        chunks.extend(await process_table(document, elements, llm))

    # image documents
    elif document.category is DocumentCategory.IMAGE:
        chunks.extend(await process_image(document, elements, llm))

    return chunks
