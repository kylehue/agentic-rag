import asyncio
import base64
import binascii
import mimetypes
from typing import Any
from uuid import uuid4

from app.llm.base import LLMProvider
from app.models.chunk import IngestedChunk, ChunkCategory
from app.models.ingestion import ProcessorPayload
from app.models.llm import LLMAttachment
from app.processors.base import Processor
from unstructured.documents.elements import Element, Image

from app.utils.string import render_template

IMAGE_PROMPT_TEMPLATE = """Describe the image for a retrieval system. Identify its subject,
important objects, people, actions, layout, visible labels, and any useful chart,
diagram, or document structure. Do not guess details that are not visible.

Nearby extracted document text is additional context only; distinguish it from
what is visibly present in the image:
{context}
"""


class ImageProcessor(Processor):
    def __init__(self, llm: LLMProvider):
        self._llm = llm

    @staticmethod
    def _text_context(elements: list[Element]) -> str:
        """Collect nearby extracted text to help LLM describe an image."""

        texts = []
        for element in elements:
            text = getattr(element, "text", None)
            if isinstance(text, str) and text.strip():
                texts.append(text.strip())

        return "\n\n".join(texts)[:8_000]

    @staticmethod
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

    @staticmethod
    def _image_bytes(
        payload: ProcessorPayload,
        image: Image,
    ) -> tuple[bytes, str, str]:
        """Get image bytes, mime type, and filename from embedded data or a file."""

        metadata = image.metadata

        # For embedded images
        embedded = ImageProcessor._decode_base64(
            getattr(metadata, "image_base64", None)
        )
        if embedded:
            mime_type = getattr(metadata, "image_mime_type", None) or "image/png"
            extension = mimetypes.guess_extension(mime_type) or ".jpg"
            return (
                embedded,
                mime_type,
                f"{uuid4()}{extension}",
            )

        # For image documents
        return (
            payload.source_bytes,
            payload.source_content_type,
            payload.source_filename,
        )

    @staticmethod
    def _description_text(description: str, context: str) -> str:
        """Build text used only to make an image retrievable by semantic search."""

        parts = [
            "Image description:",
            description or "Image description unavailable.",
        ]
        if context:
            parts.extend(("Related document text:", context))
        return "\n".join(parts)

    async def _process_single_image(
        self,
        payload: ProcessorPayload,
        image: Image,
        context: str,
    ) -> IngestedChunk:
        """Process one image and create its retrievable document chunk."""

        image_bytes, mime_type, filename = self._image_bytes(
            payload,
            image,
        )

        description = ""

        # LLM: Generate image description
        try:
            prompt = render_template(
                IMAGE_PROMPT_TEMPLATE,
                {"context": context or "(none)"},
            )

            description = await self._llm.answer(
                prompt,
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
        return IngestedChunk(
            # Avoid saving the chunk as file if it is not embedded because
            # they'll be automatically saved at the end of ingestion
            file_filename=filename if payload.is_embedded else None,
            file_bytes=image_bytes if payload.is_embedded else None,
            file_content_type=mime_type if payload.is_embedded else None,
            text=self._description_text(
                description,
                context,
            ),
            category=ChunkCategory.IMAGE,
            metadata={
                "chunk_source_id": payload.source_id,
                "chunk_source_page_number": getattr(
                    image.metadata,
                    "page_number",
                    None,
                ),
                "chunk_attach_file_to_llm": True,
            },
            orig_elements=[image],
        )

    @property
    def supported_categories(self):
        return {ChunkCategory.IMAGE}

    async def process(self, payload):
        """Create retrievable image chunks while keeping original bytes for final answers."""

        # Collect all image elements in `elements`.
        # Ideally, there should be only one image element unless it came from a text document chunk.
        images = [element for element in payload.elements if isinstance(element, Image)]

        if not images:
            return []

        # Collect text content around the image to help LLM describe it.
        # Ideally, this would be empty if the image element came from an image document.
        context = self._text_context(payload.elements)

        # Process all images concurrently.
        tasks = [
            self._process_single_image(
                payload,
                image,
                context,
            )
            for image in images
        ]

        return await asyncio.gather(*tasks)
