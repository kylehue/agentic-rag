from pathlib import Path

from app.llm.base import ChatMessage
from app.models.chunk import IngestedChunk, IngestedImageChunk, IngestedTextChunk
from app.models.content import ContentPart, ImageContent
from app.plugin.base import Plugin
from app.plugin.context import IngestionContext, IngestionFile
from app.plugin.runtime import IngestionRuntime


class ImagePlugin(Plugin):
    """Indexes images: one image chunk (embedded as an image) plus an optional
    description text chunk.

    It handles both standalone image uploads and images emitted by other
    plugins (the text plugin's extracted figures). The description is the
    user-provided source description, or, when `use_llm_description` is set,
    an LLM-generated description of the image (with the source description as
    additional context).
    """

    SUPPORTED_EXTENSIONS = {
        ".png",
        ".jpg",
        ".jpeg",
        ".webp",
        ".gif",
        ".bmp",
        ".tif",
        ".tiff",
    }

    def __init__(self, *, use_llm_description: bool = False) -> None:
        self._use_llm_description = use_llm_description

    @property
    def name(self) -> str:
        return "image"

    def accepts(self, context: IngestionContext) -> bool:
        return Path(context.file.filename).suffix.lower() in self.SUPPORTED_EXTENSIONS

    async def on_ingestion_process(
        self, context: IngestionContext, runtime: IngestionRuntime
    ) -> list[IngestedChunk]:
        # The registry only calls this for files the plugin accepts.
        file = context.file
        image = ImageContent(data=file.file_bytes, mime_type=file.content_type)
        # Both chunks share this key (the image's source id), so retrieval
        # dedups them into a single result slot.
        key = file.source_id
        chunks: list[IngestedChunk] = [
            IngestedImageChunk(plugin=self.name, image=image, key=key)
        ]

        description = await self._description(file, runtime)
        if description:
            chunks.append(
                IngestedTextChunk(plugin=self.name, text=description, key=key)
            )

        return chunks

    async def _description(
        self, file: IngestionFile, runtime: IngestionRuntime
    ) -> str | None:
        # The source description: the user-provided description for a direct
        # upload, or the caption + nearby text the text plugin computed for an
        # emitted figure.
        source = file.description

        if not self._use_llm_description:
            return source

        await runtime.report_state("describing_image")
        image = ImageContent(data=file.file_bytes, mime_type=file.content_type)
        content: list[ContentPart] = [image]
        if source:
            content.append(f"Additional context about this image: {source}")
        content.append(
            "Describe this image in detail: its subject, any visible text or "
            "labels, charts, and relevant context."
        )
        result = await runtime.llm.complete([ChatMessage(role="user", content=content)])
        text = result.content.strip()
        return text or None
