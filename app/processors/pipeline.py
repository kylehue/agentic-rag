from app.models.chunk import IngestedChunk, ChunkCategory

from app.models.ingestion import ProcessorPayload
from app.processors.table import process_table
from app.processors.text import process_text
from app.processors.image import process_image


async def process(payload: ProcessorPayload) -> list[IngestedChunk]:
    """Process document and return every searchable chunk."""
    chunks = []

    # text documents
    if payload.category is ChunkCategory.DOCUMENT:
        chunks.extend(await process_text(payload))

    # spreadsheet documents
    elif payload.category is ChunkCategory.SPREADSHEET:
        chunks.extend(await process_table(payload))

    # image documents
    elif payload.category is ChunkCategory.IMAGE:
        chunks.extend(await process_image(payload))

    return chunks
