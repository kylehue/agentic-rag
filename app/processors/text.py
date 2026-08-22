import asyncio
from dataclasses import replace

from unstructured.chunking.basic import chunk_elements
from unstructured.chunking.title import chunk_by_title
from unstructured.documents.elements import Image, Table

from app.models.chunk import IngestedChunk, ChunkCategory
from app.models.ingestion import ProcessorPayload
from app.processors.image import process_image


def _chunk_elements(payload: ProcessorPayload):
    """Run Unstructured chunking."""

    has_title = any(type(element).__name__ == "Title" for element in payload.elements)

    if has_title:
        return chunk_by_title(
            payload.elements,
            max_characters=3000,
            new_after_n_chars=2400,
            overlap=200,
            combine_text_under_n_chars=500,
        )

    return chunk_elements(
        payload.elements,
        max_characters=1500,
        overlap=200,
    )


def _embedded_elements(
    chunk: IngestedChunk,
) -> tuple[bool, bool]:
    """Check whether a document chunk contains an image or table."""

    if not chunk.orig_elements:
        return False, False

    has_table = False
    has_image = False

    for element in chunk.orig_elements:
        if isinstance(element, Image):
            has_image = True
        elif isinstance(element, Table):
            has_table = True

        if has_image and has_table:
            break

    return has_table, has_image


async def _process_image_chunk(
    payload: ProcessorPayload,
    chunk: IngestedChunk,
) -> list[IngestedChunk]:
    """Process the images contained in a document chunk."""

    if not chunk.orig_elements:
        return []

    processor_payload = replace(
        payload,
        elements=chunk.orig_elements or [],
        is_embedded=True,
    )

    return await process_image(processor_payload)


async def process_text(payload: ProcessorPayload) -> list[IngestedChunk]:
    """Split extracted text into useful chunks while preserving source elements."""

    # Chunk
    raw_chunks = await asyncio.to_thread(
        _chunk_elements,
        payload,
    )

    # Convert unstructured chunks into the app's chunk model
    document_chunks = [
        IngestedChunk(
            category=ChunkCategory.DOCUMENT,
            text=chunk.text,
            metadata={
                "chunk_source_id": payload.source_id,
                "chunk_source_page_number": getattr(
                    chunk.metadata,
                    "page_number",
                    None,
                ),
            },
            orig_elements=chunk.metadata.orig_elements or [],
        )
        for chunk in raw_chunks
    ]

    # Extract embedded tables and images
    image_tasks = []

    for chunk in document_chunks:
        # Check whether this document chunk includes an image or table element
        has_table, has_image = _embedded_elements(chunk)

        # Create dedicated image chunks when images are present
        if has_image:
            image_tasks.append(
                _process_image_chunk(
                    payload,
                    chunk,
                )
            )

        # TODO:
        # We shouldn't blindly feed a table to its processor because it could be splitted in different chunks.
        # If we want better chunking for tables in PDFs, we must collect all tables in one piece.
        # Ignoring for now. If the user expects good results for tables, they should use spreadsheet formats.

        # Create dedicated table chunks when tables are present
        # if has_table:
        #     table_chunks = await process_table(document, chunk.orig_elements)
        #     chunks.extend(table_chunks)

    # Process all embedded images concurrently.
    embedded_chunks: list[IngestedChunk] = []

    if image_tasks:
        results = await asyncio.gather(*image_tasks)

        for image_chunks in results:
            embedded_chunks.extend(image_chunks)

    document_chunks.extend(embedded_chunks)

    return document_chunks
