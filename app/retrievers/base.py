from abc import ABC, abstractmethod
from app.models.document import RetrievedDocumentChunk


class Retriever(ABC):
    @abstractmethod
    async def retrieve(self, user_query: str) -> list[RetrievedDocumentChunk]:
        """
        Retrieve chunks using the user query provided.
        Returns results in descending order (best to worst).

        Note: Higher score is better.
        """
