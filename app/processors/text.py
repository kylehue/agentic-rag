from unstructured.chunking.basic import chunk_elements
from unstructured.chunking.title import chunk_by_title
from unstructured.documents.elements import Element

from app.models.document import Document, DocumentProcessorChunk


async def process_text(
    document: Document,
    elements: list[Element],
) -> list[DocumentProcessorChunk]:
    # chunk smartly(?)
    has_title = any(type(e).__name__ == "Title" for e in elements)
    if has_title:
        chunks = chunk_by_title(
            elements,
            max_characters=3000,
            new_after_n_chars=2400,
            overlap=200,
            combine_text_under_n_chars=500,
        )
    else:
        chunks = chunk_elements(
            elements,
            max_characters=1500,
            overlap=200,
        )

    # clean chunks for output
    document_chunks = []
    for i, chunk in enumerate(chunks):
        document_chunks.append(
            DocumentProcessorChunk(
                id=f"{document.id}:{i}",
                document=document,
                text=chunk.text,
                orig_elements=chunk.metadata.orig_elements or [],
                metadata={
                    "page_number": getattr(chunk.metadata, "page_number", None),
                },
            )
        )

    return document_chunks
