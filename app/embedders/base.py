from abc import ABC, abstractmethod
from collections.abc import Sequence

from app.models.content import ImageContent


class TextEmbedder(ABC):
    """Embeds text (chunks and queries) into vectors."""

    @abstractmethod
    async def embed_text(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed text in the same order as supplied."""


class ImageEmbedder(TextEmbedder):
    """A text embedder that also embeds images, in the same shared space.

    Because text and images land in one space, a text query can be matched
    against image chunks (and vice versa). The `embed_text` here is the
    model's text encoder, used to embed queries for the image collection.
    """

    @abstractmethod
    async def embed_images(self, images: Sequence[ImageContent]) -> list[list[float]]:
        """Embed images in the same order as supplied."""
