from dataclasses import dataclass


@dataclass(frozen=True)
class ImageContent:
    """An image: its raw bytes and mime type.

    The single definition of an image part, shared by the LLM (vision
    messages), the embedders (image embeddings), and the image chunk.
    """

    data: bytes
    mime_type: str


# A message/content part: plain text or an image.
ContentPart = str | ImageContent
