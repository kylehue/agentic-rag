from collections.abc import Sequence

from app.models.document import DocumentCategory
from app.models.rag import RetrievalCandidate, RetrievedEvidence, RetrievalRequest
from app.retrievers.base import Retriever


class DocumentRetriever(Retriever):
    name = "document"

    async def retrieve(
        self,
        request: RetrievalRequest,
        candidates: Sequence[RetrievalCandidate],
    ) -> list[RetrievedEvidence]:
        """Returns text-document candidates as plain textual evidence."""
        return [
            RetrievedEvidence(
                id=item.id,
                retriever=self.name,
                document=item.document,
                content=item.text,
                metadata=item.metadata,
                score=item.score,
            )
            for item in candidates
            if item.document.category is DocumentCategory.DOCUMENT
        ]
