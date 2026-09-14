from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

from app.models.chunk import RetrievedChunk


class Retriever(ABC):
    @abstractmethod
    async def retrieve(
        self,
        user_query: str,
        where: dict[str, Any] | None = None,
    ) -> Sequence[RetrievedChunk]:
        """
        Retrieve chunks using the user query provided.
        Returns results in descending order (best to worst).

        `where` is a simple `{column: value}` equality condition map,
        applied at the index so the top_k budget is spent on the matching
        chunks rather than filtered away afterwards. None searches
        everything.

        Note: Higher score is better.
        """
