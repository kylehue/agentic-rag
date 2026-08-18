import json
from typing import Any

from app.models.document import Document, DocumentChunk


def serialize_chunk(chunk: DocumentChunk) -> str:
    """Serialize the persistent fields of a chunk without processor-only elements."""
    return json.dumps(
        {
            "id": chunk.id,
            "text": chunk.text,
            "document": chunk.document.model_dump(mode="json"),
            "metadata": chunk.metadata,
            "binary_content": (
                chunk.binary_content.hex() if chunk.binary_content is not None else None
            ),
            "binary_mime_type": chunk.binary_mime_type,
        },
        default=str,
    )


def deserialize_chunk(chunk_json: str) -> DocumentChunk:
    """Deserialize a chunk record stored by :func:`serialize_chunk`."""
    try:
        payload: dict[str, Any] = json.loads(chunk_json)
        binary_content = payload.get("binary_content")
        return DocumentChunk(
            id=str(payload["id"]),
            text=str(payload["text"]),
            document=Document.model_validate(payload["document"]),
            metadata=dict(payload.get("metadata", {})),
            binary_content=(
                bytes.fromhex(binary_content)
                if isinstance(binary_content, str)
                else None
            ),
            binary_mime_type=(
                str(payload["binary_mime_type"])
                if payload.get("binary_mime_type") is not None
                else None
            ),
        )
    except (
        AttributeError,
        KeyError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ) as error:
        raise ValueError("Invalid stored chunk JSON") from error
