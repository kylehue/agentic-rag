import asyncio
from collections.abc import Sequence
from io import BytesIO

from fastembed import ImageEmbedding, TextEmbedding

from app.embedders.base import ImageEmbedder
from app.models.content import ImageContent


def _to_file_objects(images: Sequence[ImageContent]) -> list[BytesIO]:
    """fastembed's image embedder opens each input as a file, so wrap the raw
    bytes in file objects."""
    return [BytesIO(image.data) for image in images]


class ClipImageEmbedder(ImageEmbedder):
    """Multimodal CLIP embeddings via fastembed, in a single shared space.

    The text side is a fastembed `TextEmbedding` (the CLIP text model) and the
    vision side a fastembed `ImageEmbedding` (the CLIP vision model); because
    both encode into the same CLIP space, a text query matches image chunks.
    Both models are loaded eagerly (download + load at construction, into
    `cache_dir`), and fastembed is synchronous, so inference runs in worker
    threads.
    """

    def __init__(
        self,
        *,
        text_model: str,
        image_model: str,
        batch_size: int = 32,
        cache_dir: str | None = None,
    ) -> None:
        self._batch_size = batch_size
        self._text_model = TextEmbedding(
            text_model, cache_dir=cache_dir, lazy_load=False
        )
        self._image_model = ImageEmbedding(
            image_model, cache_dir=cache_dir, lazy_load=False
        )

    async def embed_text(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        if any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError("Cannot embed empty text")

        vectors = await asyncio.to_thread(
            list, self._text_model.embed(list(texts), self._batch_size)
        )
        embeddings = [vector.tolist() for vector in vectors]
        if len(embeddings) != len(texts):
            raise RuntimeError(
                "Embedding provider returned an unexpected number of vectors"
            )
        return embeddings

    async def embed_images(
        self, images: Sequence[ImageContent]
    ) -> list[list[float]]:
        if not images:
            return []

        # fastembed opens each input as a file; the stub types it as
        # str/Path/PIL only, but file objects work at runtime.
        vectors = await asyncio.to_thread(
            list,
            self._image_model.embed(
                _to_file_objects(images), self._batch_size  # type: ignore[arg-type]
            ),
        )
        embeddings = [vector.tolist() for vector in vectors]
        if len(embeddings) != len(images):
            raise RuntimeError(
                "Embedding provider returned an unexpected number of vectors"
            )
        return embeddings
