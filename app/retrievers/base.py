from abc import ABC, abstractmethod
from collections.abc import Sequence

from app.models.rag import RetrievalCandidate, RetrievedEvidence, RetrievalRequest


class Retriever(ABC):
    """Interface for turning search candidates into answer-ready evidence."""

    name: str

    @abstractmethod
    async def retrieve(
        self,
        request: RetrievalRequest,
        candidates: Sequence[RetrievalCandidate],
    ) -> list[RetrievedEvidence]:
        """Return evidence. External retrievers may ignore vector candidates."""
