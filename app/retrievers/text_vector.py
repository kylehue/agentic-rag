from app.database import CHUNK_TABLE_NAME
from app.embedders.base import TextEmbedder
from app.models.chunk import RetrievedTextChunk
from app.retrievers.base import Retriever
from app.store_sql.base import SqlStorage
from app.store_vector.base import VectorStorage


class TextVectorRetriever(Retriever):
    def __init__(
        self,
        embedder: TextEmbedder,
        text_vector_storage: VectorStorage,
        sql_storage: SqlStorage,
        top_k: int,
    ):
        self._embedder = embedder
        self._text_vector_storage = text_vector_storage
        self._sql_storage = sql_storage
        self._top_k = top_k

    async def retrieve(
        self,
        user_query: str,
        where: dict | None = None,
    ) -> list[RetrievedTextChunk]:
        # Retrieve from the vector database. The where map is the index's
        # own metadata filter (the chunks' stored metadata, e.g. their chat).
        query_embedding = (await self._embedder.embed_text([user_query]))[0]

        vector_results = await self._text_vector_storage.search(
            query_embedding,
            self._top_k,
            where=where,
        )

        if not vector_results:
            return []

        # Chroma's order is authoritative for vector ranking.
        chunk_ids = [result.chunk_id for result in vector_results]

        # SQL IN (...) does NOT guarantee the same order.
        raw_chunks = await self._sql_storage.get_all(
            CHUNK_TABLE_NAME,
            lambda table: table.c.chunk_id.in_(chunk_ids),
        )

        # Map database rows by application-level chunk ID.
        chunks_by_id = {raw_chunk["chunk_id"]: raw_chunk for raw_chunk in raw_chunks}

        # Restore the original vector ranking order and attach scores.
        results: list[RetrievedTextChunk] = []

        for vector_result in vector_results:
            raw_chunk = chunks_by_id.get(vector_result.chunk_id)

            if raw_chunk is None:
                continue

            results.append(
                RetrievedTextChunk.from_dict(
                    raw_chunk,
                    score=vector_result.score,
                    text=raw_chunk["text"],
                )
            )

        return results
