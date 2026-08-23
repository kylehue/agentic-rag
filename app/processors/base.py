from abc import ABC, abstractmethod
from collections.abc import Sequence

from app.models.chunk import ChunkCategory, IngestedChunk
from app.models.ingestion import ProcessorPayload


class Processor(ABC):
    @property
    @abstractmethod
    def supported_categories(self) -> set[ChunkCategory]:
        """Categories this processor can process."""

    @abstractmethod
    async def process(
        self,
        payload: ProcessorPayload,
    ) -> Sequence[IngestedChunk]:
        """Process a source file into chunks."""
