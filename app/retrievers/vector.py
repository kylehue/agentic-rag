from app.core.config import settings
from app.embedders.base import Embedder
from app.models.chunk import RetrievedChunk
from app.retrievers.base import Retriever
from app.store_sql.base import SqlStorage
from app.store_vector.base import VectorStorage


class VectorRetriever(Retriever):
    def __init__(
        self,
        embedder: Embedder,
        vector_storage: VectorStorage,
        sql_storage: SqlStorage,
        top_k: int,
    ):
        self._embedder = embedder
        self._vector_storage = vector_storage
        self._sql_storage = sql_storage
        self._top_k = top_k

    async def retrieve(self, user_query: str):
        # Retrieve from vector database
        query_embedding = await self._embedder.embed_query(user_query)
        vector_results = await self._vector_storage.search(
            query_embedding,
            self._top_k,
        )
        if not vector_results:
            return []

        chunk_ids = [result.id for result in vector_results]

        # Map vector results to chunk data
        table = await self._sql_storage.get_table(settings.CHUNK_TABLE_NAME)
        raw_chunks = await self._sql_storage.get_all(
            settings.CHUNK_TABLE_NAME,
            [table.c.chunk_id.in_(chunk_ids)],
        )

        chunks_by_id = {chunk["chunk_id"]: chunk for chunk in raw_chunks}

        # Restore vector-search order and attach scores
        results: list[RetrievedChunk] = []

        for vector_result in vector_results:
            chunk = chunks_by_id.get(vector_result.id)

            if chunk is None:
                continue

            results.append(
                RetrievedChunk(
                    **chunk,
                    score=vector_result.score,
                )
            )

        return results
