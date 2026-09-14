import asyncio
from pathlib import Path
from typing import Any, Sequence

import chromadb

from app.models.vector import VectorSearchResult
from app.store_vector.base import VectorStorage


class LocalVectorStorage(VectorStorage):
    """Stores chunk IDs and embeddings in local Chroma."""

    def __init__(
        self,
        storage_dir: str | Path,
        collection_name: str,
    ):
        self._storage_dir = Path(storage_dir)
        self._storage_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(self._storage_dir))
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    async def close(self) -> None:
        pass

    async def add(
        self,
        ids: Sequence[str],
        embeddings: Sequence[Sequence[float]],
        metadatas: Sequence[dict[str, Any]] | None = None,
    ) -> None:
        if len(ids) != len(embeddings):
            raise ValueError("ids and embeddings must have the same length")
        if not ids:
            return
        if metadatas is not None and len(metadatas) != len(ids):
            raise ValueError("ids and metadatas must have the same length")

        # Delete first so re-indexing an existing ID also removes legacy Chroma
        # documents and metadata from versions that stored chunk payloads here.
        vectors: list[Sequence[float] | Sequence[int]] = [
            list(embedding) for embedding in embeddings
        ]
        await asyncio.to_thread(self._collection.delete, ids=list(ids))
        await asyncio.to_thread(
            self._collection.upsert,
            ids=list(ids),
            embeddings=vectors,
            metadatas=list(metadatas) if metadatas is not None else None,
        )

    async def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        where: dict[str, Any] | None = None,
    ) -> list[VectorSearchResult]:
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        collection_count = await asyncio.to_thread(self._collection.count)
        if collection_count == 0:
            return []

        query_vectors: list[Sequence[float] | Sequence[int]] = [query_embedding]
        result = await asyncio.to_thread(
            self._collection.query,
            query_embeddings=query_vectors,
            n_results=min(top_k, collection_count),
            where=where,
            include=["distances"],
        )
        ids = result["ids"][0]
        distances = (result["distances"] or [])[0]
        return [
            VectorSearchResult(
                chunk_id=chunk_id,
                score=1 - distance,  # in cosine similarity, lower is better
            )
            for chunk_id, distance in zip(ids, distances)
        ]

    async def delete(self, ids: Sequence[str]) -> None:
        if ids:
            await asyncio.to_thread(self._collection.delete, ids=list(ids))
