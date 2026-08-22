from abc import ABC, abstractmethod
from collections.abc import Sequence
from app.models.chunk import RetrievedChunk


class Retriever(ABC):
    @abstractmethod
    async def retrieve(
        self,
        user_query: str,
    ) -> Sequence[RetrievedChunk]:
        """
        Retrieve chunks using the user query provided.
        Returns results in descending order (best to worst).

        Note: Higher score is better.
        """
