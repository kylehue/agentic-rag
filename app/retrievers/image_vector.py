from app.database import CHUNK_TABLE_NAME
from app.embedders.base import ImageEmbedder
from app.models.chunk import RetrievedImageChunk
from app.retrievers.base import Retriever
from app.store_sql.base import SqlStorage
from app.store_vector.base import VectorStorage


class ImageVectorRetriever(Retriever):
    """Retrieves image chunks by matching the text query (embedded with the
    image embedder's text encoder) against the image vector collection."""

    def __init__(
        self,
        embedder: ImageEmbedder,
        image_vector_storage: VectorStorage,
        sql_storage: SqlStorage,
        top_k: int,
    ):
        self._embedder = embedder
        self._image_vector_storage = image_vector_storage
        self._sql_storage = sql_storage
        self._top_k = top_k

    async def retrieve(
        self,
        user_query: str,
        where: dict | None = None,
    ) -> list[RetrievedImageChunk]:
        # The image embedder's text encoder places the query in the image
        # space, so it matches the image chunks' vectors.
        query_embedding = (await self._embedder.embed_text([user_query]))[0]

        vector_results = await self._image_vector_storage.search(
            query_embedding,
            self._top_k,
            where=where,
        )

        if not vector_results:
            return []

        chunk_ids = [result.chunk_id for result in vector_results]
        raw_chunks = await self._sql_storage.get_all(
            CHUNK_TABLE_NAME,
            lambda table: table.c.chunk_id.in_(chunk_ids),
        )
        chunks_by_id = {
            raw_chunk["chunk_id"]: raw_chunk for raw_chunk in raw_chunks
        }

        results: list[RetrievedImageChunk] = []
        for vector_result in vector_results:
            raw_chunk = chunks_by_id.get(vector_result.chunk_id)
            if raw_chunk is None:
                continue
            # The image bytes are not loaded here; they are fetched on demand
            # via source_id (see the view_images tool).
            results.append(
                RetrievedImageChunk.from_dict(raw_chunk, score=vector_result.score)
            )

        return results
