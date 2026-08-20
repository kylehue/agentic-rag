from app.models.document import Document, DocumentCategory

from app.models.ingestion import ProcessorPayload
from app.processors.table import process_table
from app.processors.text import process_text
from app.processors.image import process_image


async def process(payload: ProcessorPayload) -> list[Document]:
    """Process document and return every searchable chunk."""
    chunks = []

    # text documents
    if payload.document_category is DocumentCategory.DOCUMENT:
        chunks.extend(await process_text(payload))

    # spreadsheet documents
    elif payload.document_category is DocumentCategory.SPREADSHEET:
        chunks.extend(await process_table(payload))

    # image documents
    elif payload.document_category is DocumentCategory.IMAGE:
        chunks.extend(await process_image(payload))

    return chunks
