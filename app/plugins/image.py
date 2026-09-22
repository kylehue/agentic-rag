import logging
from pathlib import Path

from app.llm.base import ChatMessage, LLMProvider
from app.models.chunk import IngestedChunk, IngestedImageChunk, IngestedTextChunk
from app.models.content import ContentPart, ImageContent
from app.plugin.base import Plugin
from app.plugin.context import IngestionContext, IngestionFile
from app.plugin.runtime import IngestionRuntime

logger = logging.getLogger(__name__)


class ImagePlugin(Plugin):
    """Indexes images: one image chunk (embedded as an image) plus a
    description text chunk.

    It handles both standalone image uploads and images emitted by other
    plugins (the text plugin's extracted figures). When `use_llm_description`
    is set (the default), every image gets an LLM description using the
    user-provided source description as additional context; otherwise only
    the source description is used. The plugin uses its own LLM when one is
    provided (the intended wiring is a separate cheap vision model) and falls
    back to the pipeline LLM otherwise. A description that fails degrades to
    the source description, so a flaky model never blocks ingestion.
    """

    SUPPORTED_EXTENSIONS = frozenset(
        {
            ".png",
            ".jpg",
            ".jpeg",
            ".webp",
            ".gif",
            ".bmp",
            ".tif",
            ".tiff",
        }
    )

    def __init__(
        self,
        *,
        use_llm_description: bool = True,
        llm: LLMProvider | None = None,
    ) -> None:
        self._use_llm_description = use_llm_description
        self._llm = llm

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
        source = file.description
        if not self._use_llm_description:
            return source
        await runtime.report_state("describing_image")
        try:
            return await self._describe(file, source, self._llm or runtime.llm)
        except Exception:
            logger.warning(
                "Image description failed for '%s'; using the source "
                "description instead.",
                file.filename,
                exc_info=True,
            )
            return source

    @staticmethod
    async def _describe(
        file: IngestionFile, source: str | None, llm: LLMProvider
    ) -> str | None:
        # The LLM description of the image, with the source description (if
        # any) as additional context. Falls back to the source when the model
        # returns nothing.
        image = ImageContent(data=file.file_bytes, mime_type=file.content_type)
        content: list[ContentPart] = [image]
        if source:
            content.append(f"Additional context about this image: {source}")
        content.append(
            "Describe this image in detail: its subject, any visible text or "
            "labels, charts, and relevant context."
        )
        result = await llm.complete([ChatMessage(role="user", content=content)])
        text = result.content.strip()
        return text or source
