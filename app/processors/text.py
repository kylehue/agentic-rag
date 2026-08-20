from unstructured.chunking.basic import chunk_elements
from unstructured.chunking.title import chunk_by_title
from unstructured.documents.elements import Element, Image, Table

from app.llm.base import LLMProvider
from app.models.document import Document
from app.processors.image import process_image


async def process_text(
    document: Document,
    elements: list[Element],
    llm: LLMProvider,
) -> list[Document]:
    """Split extracted text into useful chunks while preserving source elements."""

    # Chunk
    has_title = any(type(e).__name__ == "Title" for e in elements)
    if has_title:
        raw_chunks = chunk_by_title(
            elements,
            max_characters=3000,
            new_after_n_chars=2400,
            overlap=200,
            combine_text_under_n_chars=500,
        )
    else:
        raw_chunks = chunk_elements(
            elements,
            max_characters=1500,
            overlap=200,
        )

    # Convert unstructured chunks into the app's chunk model
    document_chunks: list[Document] = []
    for i, chunk in enumerate(raw_chunks):
        document_chunks.append(
            Document(
                **document.model_dump(),
                id=f"{document.id}:{i}",
                text=chunk.text,
                orig_elements=chunk.metadata.orig_elements or [],
                metadata={
                    "page_number": getattr(chunk.metadata, "page_number", None),
                },
            )
        )

    # Extract tables and images
    for chunk in document_chunks:
        # Check whether this document chunk includes an image or table element
        if chunk.orig_elements is None:
            continue
        has_table, has_image = False, False
        for element in chunk.orig_elements:
            if isinstance(element, Image):
                has_image = True
            elif isinstance(element, Table):
                has_table = True
            if has_image and has_table:
                break

        # Create dedicated image chunks when images are present
        if has_image:
            image_chunks = await process_image(document, chunk.orig_elements, llm)
            document_chunks.extend(image_chunks)

        # TODO:
        # We shouldn't blindly feed a table to its processor because it could be splitted in different chunks.
        # If we want better chunking for tables in PDFs, we must collect all tables in one piece.
        # Ignoring for now. If the user expects good results for tables, they should use spreadsheet formats.

        # Create dedicated table chunks when tables are present
        # if has_table:
        #     table_chunks = await process_table(document, chunk.orig_elements)
        #     chunks.extend(table_chunks)

    return document_chunks
