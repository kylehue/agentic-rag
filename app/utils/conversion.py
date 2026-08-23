import base64
import binascii
from typing import Any

from app.core.file_types import IMAGE_TYPES, SPREADSHEET_TYPES
from app.models.chunk import ChunkCategory


def filename_to_chunk_category(filename: str) -> ChunkCategory:
    normalized = filename.split(".")[-1]
    if normalized in SPREADSHEET_TYPES:
        return ChunkCategory.SPREADSHEET
    elif normalized in IMAGE_TYPES:
        return ChunkCategory.IMAGE
    else:
        return ChunkCategory.DOCUMENT


def base64_to_bytes(value: Any) -> bytes | None:
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
