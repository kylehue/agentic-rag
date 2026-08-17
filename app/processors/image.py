import base64
import binascii
import mimetypes
from pathlib import Path
from typing import Any

from app.models.llm import LLMAttachment
from app.llm.base import LLMProvider
from app.models.document import Document, DocumentChunk
from unstructured.documents.elements import Element, Image

IMAGE_PROMPT = """Describe the image for a retrieval system. Identify its subject,
important objects, people, actions, layout, visible labels, and any useful chart,
diagram, or document structure. Do not guess details that are not visible.

Nearby extracted document text is additional context only; distinguish it from
what is visibly present in the image:
{context}
"""


def _text_context(elements: list[Element]) -> str:
    """Collect nearby extracted text to help describe an image during ingestion."""
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


def _image_bytes(document: Document, image: Image) -> tuple[bytes | None, str, str]:
    """Get image bytes from embedded data or a file, and record where they came from."""
    metadata = image.metadata
    embedded = _decode_base64(getattr(metadata, "image_base64", None))
    if embedded:
        mime_type = getattr(metadata, "image_mime_type", None) or "image/png"
        return embedded, mime_type, "embedded"

    candidates = [getattr(metadata, "image_path", None), document.path]
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate)
        if path.is_file():
            try:
                payload = path.read_bytes()
            except OSError:
                continue
            mime_type = (
                getattr(metadata, "image_mime_type", None)
                or mimetypes.guess_type(path.name)[0]
                or "application/octet-stream"
            )
            return payload, mime_type, "file"
    return (
        None,
        getattr(metadata, "image_mime_type", None) or "image/png",
        "unavailable",
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
) -> list[DocumentChunk]:
    """Create retrievable image chunks while keeping original bytes for final answers."""
    images = [element for element in elements if isinstance(element, Image)]
    if not images:
        return []

    context = _text_context(elements)
    chunks: list[DocumentChunk] = []
    for i, image in enumerate(images):
        image_bytes, mime_type, source = _image_bytes(document, image)
        description = ""
        if image_bytes:
            try:
                description = await llm.answer(
                    IMAGE_PROMPT.format(context=context or "(none)"),
                    attachments=(LLMAttachment(image_bytes, mime_type),),
                )
            except Exception:
                pass

        chunks.append(
            DocumentChunk(
                id=f"{document.id}:image:{i}",
                document=document,
                text=_description_text(description, context),
                orig_elements=[image],
                metadata={
                    "page_number": getattr(image.metadata, "page_number", None),
                    "image_mime_type": mime_type,
                    "image_source": source,
                    "description": description,
                    "context": context,
                },
                binary_content=image_bytes,
                binary_mime_type=mime_type if image_bytes else None,
            )
        )
    return chunks
