from abc import ABC, abstractmethod
from collections.abc import Sequence

from app.models.chunk import RetrievedChunk


class Reranker(ABC):
    """Re-ranks retrieved chunks by relevance to a query.

    Sits after the retriever (which produces the candidate set) and before
    the plugins' finalization: it re-scores and reorders the candidates. It
    does not add or drop chunks, so the candidate set is preserved and only
    its ranking and scores change.
    """

    @abstractmethod
    async def rerank(
        self,
        query: str,
        chunks: Sequence[RetrievedChunk],
    ) -> list[RetrievedChunk]:
        """Return the chunks re-scored and ordered by relevance to `query`.

        The result is the same set of chunks in descending order (best to
        worst), with each chunk's `score` replaced by the reranker's
        relevance score (higher is better, consistent with the retrievers).
        """
