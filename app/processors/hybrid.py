from collections.abc import Sequence

from app.models.chunk import ChunkCategory, IngestedChunk
from app.models.ingestion import ProcessorPayload
from app.processors.base import Processor


class HybridProcessor(Processor):
    def __init__(self, processors: Sequence[Processor]):
        self._processors = processors

    @property
    def supported_categories(self) -> set[ChunkCategory]:
        return {
            category
            for processor in self._processors
            for category in processor.supported_categories
        }

    async def process(
        self,
        payload: ProcessorPayload,
    ) -> list[IngestedChunk]:
        """Route the payload to processors that support its category."""

        chunks: list[IngestedChunk] = []

        for processor in self._processors:
            if payload.category in processor.supported_categories:
                chunks.extend(await processor.process(payload))

        return chunks
