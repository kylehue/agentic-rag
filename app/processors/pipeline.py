from app.models.document import Document, DocumentCategory, DocumentProcessorChunk

from app.processors.table import process_table
from app.processors.text import process_text
from app.processors.image import process_image
from unstructured.documents.elements import Element, Table, Image


async def process(
    document: Document,
    elements: list[Element],
) -> list[DocumentProcessorChunk]:
    """Processor for all kinds of documents."""
    chunks = []

    # text documents
    if document.category is DocumentCategory.DOCUMENT:
        text_chunks = await process_text(document, elements)
        chunks.extend(text_chunks)

        # chunk sub-processing for image/tables found in docs
        for chunk in text_chunks:
            # check if it has image or table
            has_table, has_image = False, False
            for element in chunk.orig_elements:
                if isinstance(element, Image):
                    has_image = True
                elif isinstance(element, Table):
                    has_table = True
                if has_image and has_table:
                    break

            # sub-process for every chunk that has image or table
            if has_image:
                image_chunks = await process_image(document, chunk.orig_elements)
                chunks.extend(image_chunks)
            """
            We shouldn't blindly feed a table to its processor because it could be splitted in different chunks.
            If we want better chunking for tables in PDFs, we must pre-process them.
            Pre-processing such as merging splitted tables.
            Ignoring for now. If the user expects good results for tables, they should use spreadsheet formats.
            """
            # if has_table:
            #     table_chunks = await process_table(document, chunk.orig_elements)
            #     chunks.extend(table_chunks)

    # spreadsheet documents
    elif document.category is DocumentCategory.SPREADSHEET:
        chunks.extend(await process_table(document, elements))

    # image documents
    elif document.category is DocumentCategory.IMAGE:
        chunks.extend(await process_image(document, elements))

    return chunks
