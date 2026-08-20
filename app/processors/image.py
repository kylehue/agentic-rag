import base64
import binascii
import mimetypes
from typing import Any

from app.models.llm import LLMAttachment
from app.llm.base import LLMProvider
from app.models.document import Document
from unstructured.documents.elements import Element, Image
from app.dependencies import document_service

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


async def _image_bytes(document: Document, image: Image) -> tuple[bytes | None, str]:
    """Get image bytes from embedded data or a file, and record where they came from."""

    metadata = image.metadata

    # For images in text documents
    embedded = _decode_base64(getattr(metadata, "image_base64", None))
    if embedded:
        mime_type = getattr(metadata, "image_mime_type", None) or "image/png"
        return embedded, mime_type

    # For image documents
    try:
        payload = await document_service.read_bytes(document.id)
        mime_type = (
            getattr(metadata, "image_mime_type", None)
            or mimetypes.guess_type(document.path)[0]
            or "application/octet-stream"
        )
        return payload, mime_type
    except:
        return (
            None,
            getattr(metadata, "image_mime_type", None) or "image/png",
        )


def _description_text(description: str, context: str) -> str:
    """Build text used only to make an image retrievable by semantic search."""

    parts = ["Image description:", description or "Image description unavailable."]
    if context:
        parts.extend(("Related document text:", context))
    return "\n".join(parts)


async def process_image(
    document: Document,
    elements: list[Element],
    llm: LLMProvider,
) -> list[Document]:
    """Create retrievable image chunks while keeping original bytes for final answers."""

    # Collect all image elements in `elements`.
    # Ideally, there should be only one image element unless it came from a text document chunk.
    images = [e for e in elements if isinstance(e, Image)]
    if not images:
        return []

    # Collect text content around the image to help LLM describe the image.
    # Ideally, this would be empty if the image element came from an image document.
    context = _text_context(elements)

    chunks: list[Document] = []
    for i, image in enumerate(images):
        image_bytes, mime_type = await _image_bytes(document, image)
        description = ""

        # LLM: Generate image description
        if image_bytes:
            try:
                description = await llm.answer(
                    IMAGE_PROMPT.format(context=context or "(none)"),
                    attachments=(LLMAttachment(image_bytes, mime_type),),
                )
            except Exception:
                pass

        # Output chunk
        chunks.append(
            Document(
                **document.model_dump(),
                id=f"{document.id}:image:{i}",
                text=_description_text(description, context),
                orig_elements=[image],
                metadata={
                    "page_number": getattr(image.metadata, "page_number", None),
                    "is_image_embedded": bool(
                        getattr(image.metadata, "image_base64", None)
                    ),
                },
                binary_content=image_bytes,
                binary_mime_type=mime_type if image_bytes else None,
            )
        )

    return chunks
