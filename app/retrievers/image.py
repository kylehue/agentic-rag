from collections.abc import Sequence

from app.models.document import DocumentCategory
from app.models.rag import RetrievalCandidate, RetrievedEvidence, RetrievalRequest
from app.retrievers.base import Retriever


class ImageRetriever(Retriever):
    name = "image"

    async def retrieve(
        self,
        request: RetrievalRequest,
        candidates: Sequence[RetrievalCandidate],
    ) -> list[RetrievedEvidence]:
        return [
            RetrievedEvidence(
                id=item.id,
                retriever=self.name,
                document=item.document,
                content="Image evidence is attached for multimodal answer synthesis.",
                metadata=item.metadata,
                score=item.score,
                # for llm
                binary_content=item.binary_content,
                binary_mime_type=item.binary_mime_type,
            )
            for item in candidates
            if item.document.category is DocumentCategory.IMAGE
        ]
