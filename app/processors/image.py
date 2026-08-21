import asyncio
import base64
import binascii
import mimetypes
from typing import Any

from app.models.document import Document, DocumentCategory
from app.models.ingestion import ProcessorPayload
from app.models.llm import LLMAttachment
from unstructured.documents.elements import Element, Image

IMAGE_PROMPT = """Describe the image for a retrieval system. Identify its subject,
important objects, people, actions, layout, visible labels, and any useful chart,
diagram, or document structure. Do not guess details that are not visible.

Nearby extracted document text is additional context only; distinguish it from
what is visibly present in the image:
{context}
"""


def _text_context(elements: list[Element]) -> str:
    """Collect nearby extracted text to help LLM describe an image."""

    texts = []
    for element in elements:
        text = getattr(element, "text", None)
        if isinstance(text, str) and text.strip():
            texts.append(text.strip())

    return "\n\n".join(texts)[:8_000]


def _decode_base64(value: Any) -> bytes | None:
    """Decode a base64 image payload safely, returning None for invalid input."""

    if isinstance(value, bytes):
        return value
    if not isinstance(value, str) or not value.strip():
        return None
    encoded = value.strip()
    if encoded.startswith("data:") and "," in encoded:
        encoded = encoded.split(",", 1)[1]
    try:
        return base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError):
        return None


def _image_bytes(
    payload: ProcessorPayload,
    image: Image,
) -> tuple[bytes, str, str]:
    """Get image bytes, mime type, and filename from embedded data or a file."""

    metadata = image.metadata

    # For embedded images
    embedded = _decode_base64(getattr(metadata, "image_base64", None))
    if embedded:
        mime_type = getattr(metadata, "image_mime_type", None) or "image/png"
        extension = mimetypes.guess_extension(mime_type) or ".jpg"
        return (
            embedded,
            mime_type,
            payload.file_filename + extension,
        )

    # For image documents
    return (
        payload.file_bytes,
        payload.file_content_type,
        payload.file_filename,
    )


def _description_text(description: str, context: str) -> str:
    """Build text used only to make an image retrievable by semantic search."""

    parts = ["Image description:", description or "Image description unavailable."]
    if context:
        parts.extend(("Related document text:", context))
    return "\n".join(parts)


async def _process_single_image(
    payload: ProcessorPayload,
    image: Image,
    context: str,
) -> Document:
    """Process one image and create its retrievable document chunk."""

    image_bytes, mime_type, filename = _image_bytes(
        payload,
        image,
    )

    description = ""

    # LLM: Generate image description
    try:
        description = await payload.llm.answer(
            IMAGE_PROMPT.format(
                context=context or "(none)",
            ),
            attachments=(
                LLMAttachment(
                    image_bytes,
                    mime_type,
                ),
            ),
        )
    except Exception:
        pass

    # Output chunk
    return Document(
        file_filename=filename,
        file_bytes=image_bytes,
        file_content_type=mime_type,
        text=_description_text(
            description,
            context,
        ),
        category=DocumentCategory.IMAGE,
        metadata={
            "file_id": payload.file_id,
            "page_number": getattr(
                image.metadata,
                "page_number",
                None,
            ),
        },
        orig_elements=[image],
    )


async def process_image(payload: ProcessorPayload) -> list[Document]:
    """Create retrievable image chunks while keeping original bytes for final answers."""

    # Collect all image elements in `elements`.
    # Ideally, there should be only one image element unless it came from a text document chunk.
    images = [element for element in payload.elements if isinstance(element, Image)]

    if not images:
        return []

    # Collect text content around the image to help LLM describe it.
    # Ideally, this would be empty if the image element came from an image document.
    context = _text_context(payload.elements)

    # Process all images concurrently.
    tasks = [
        _process_single_image(
            payload,
            image,
            context,
        )
        for image in images
    ]

    return await asyncio.gather(*tasks)
