import asyncio
from pathlib import Path
from typing import Sequence

import chromadb

from app.core.config import settings
from app.store_vector.base import VectorStorage


class LocalVectorStorage(VectorStorage):
    """Stores chunk IDs and embeddings in local Chroma."""

    def __init__(
        self,
        storage_dir: str | Path | None = None,
        collection_name: str | None = None,
    ):
        path = Path(storage_dir or settings.VECTOR_LOCAL_STORAGE_DIR)
        path.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(path))
        self._collection = self._client.get_or_create_collection(
            name=collection_name or settings.VECTOR_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    async def add(
        self,
        ids: Sequence[str],
        embeddings: Sequence[Sequence[float]],
    ) -> None:
        """Upsert chunk IDs and their matching embeddings into Chroma."""
        if len(ids) != len(embeddings):
            raise ValueError("ids and embeddings must have the same length")
        if not ids:
            return

        # Delete first so re-indexing an existing ID also removes legacy Chroma
        # documents and metadata from versions that stored chunk payloads here.
        await asyncio.to_thread(self._collection.delete, ids=list(ids))
        await asyncio.to_thread(
            self._collection.upsert,
            ids=list(ids),
            embeddings=[list(embedding) for embedding in embeddings],
        )

    async def search(self, query_embedding: list[float], top_k: int = 5) -> list[str]:
        """Return the IDs of the nearest chunks in ranked order."""
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        collection_count = await asyncio.to_thread(self._collection.count)
        if collection_count == 0:
            return []

        result = await asyncio.to_thread(
            self._collection.query,
            query_embeddings=[query_embedding],
            n_results=min(top_k, collection_count),
            include=[],
        )
        return result["ids"][0]

    async def delete(self, ids: Sequence[str]) -> None:
        """Remove vectors for the supplied chunk IDs."""
        if ids:
            await asyncio.to_thread(self._collection.delete, ids=list(ids))
