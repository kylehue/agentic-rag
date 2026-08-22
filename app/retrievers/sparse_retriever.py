from app.core.config import settings
from app.models.document import RetrievedDocumentChunk
from app.retrievers.base import Retriever
from app.store_sql2.base import SqlStorage


class SparseRetriever(Retriever):
    def __init__(
        self,
        sql_storage: SqlStorage,
        top_k: int,
    ):
        self._sql_storage = sql_storage
        self._top_k = top_k

    async def retrieve(self, user_query):
        raw_chunks = await self._sql_storage.search(
            settings.CHUNK_TABLE_NAME, user_query, self._top_k
        )

        result = []
        for i, raw_chunk in enumerate(raw_chunks, 1):
            result.append(
                RetrievedDocumentChunk(
                    **raw_chunk,
                    score=0,  # doesn't matter for sparse search (as long as it's sorted)
                )
            )

        return result
