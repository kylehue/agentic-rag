import asyncio
import json
from pathlib import Path
from typing import Any, Sequence

import chromadb

from app.core.config import settings
from app.models.document import Document, DocumentChunk
from app.store_vector.base import VectorStorage


class LocalVectorStorage(VectorStorage):
    """A persistent Chroma implementation of the provider-neutral vector API."""

    def __init__(
        self, storage_dir: str | Path | None = None, collection_name: str | None = None
    ):
        path = Path(storage_dir or settings.VECTOR_LOCAL_STORAGE_DIR)
        path.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(path))
        self._collection = self._client.get_or_create_collection(
            name=collection_name or settings.VECTOR_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    @staticmethod
    def _metadata(chunk: DocumentChunk) -> dict[str, str | int | float | bool]:
        """Chroma metadata is scalar-only; retain richer processor metadata as JSON."""
        metadata: dict[str, str | int | float | bool] = {
            "document_id": chunk.document.id,
            "filename": chunk.document.filename,
            "category": chunk.document.category.value,
            "document": json.dumps(chunk.document.model_dump(mode="json")),
            "chunk_metadata": json.dumps(chunk.metadata, default=str),
        }
        page_number = chunk.metadata.get("page_number")
        if isinstance(page_number, (str, int, float, bool)):
            metadata["page_number"] = page_number
        return metadata

    async def add_documents(
        self,
        documents: Sequence[DocumentChunk],
        embeddings: Sequence[Sequence[float]],
    ) -> None:
        if len(documents) != len(embeddings):
            raise ValueError("documents and embeddings must have the same length")
        if not documents:
            return

        await asyncio.to_thread(
            self._collection.upsert,
            ids=[chunk.id for chunk in documents],
            documents=[chunk.text for chunk in documents],
            embeddings=[list(embedding) for embedding in embeddings],
            metadatas=[self._metadata(chunk) for chunk in documents],
        )

    async def search(
        self, query_embedding: list[float], top_k: int = 5
    ) -> list[DocumentChunk]:
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        collection_count = await asyncio.to_thread(self._collection.count)
        if collection_count == 0:
            return []

        result = await asyncio.to_thread(
            self._collection.query,
            query_embeddings=[query_embedding],
            n_results=min(top_k, collection_count),
            include=["documents", "metadatas", "distances"],
        )

        return [
            self._chunk_from_result(identifier, text, metadata, distance)  # type: ignore
            for identifier, text, metadata, distance in zip(
                result["ids"][0],
                result["documents"][0],  # type: ignore
                result["metadatas"][0],  # type: ignore
                result["distances"][0],  # type: ignore
            )
        ]

    @classmethod
    def _chunk_from_result(
        cls,
        identifier: str,
        text: str,
        metadata: dict[str, Any] | None,
        distance: float | None,
    ) -> DocumentChunk:
        stored_metadata = dict(metadata or {})
        raw_document = stored_metadata.get("document")
        try:
            document = Document.model_validate_json(raw_document)  # type: ignore
        except (TypeError, ValueError) as error:
            raise ValueError(
                "Stored vector is missing a valid document payload; re-ingest the document."
            ) from error

        chunk_metadata = cls._decode_metadata(stored_metadata)
        chunk_metadata["vector_distance"] = distance
        return DocumentChunk(
            id=identifier,
            text=text,
            document=document,
            metadata=chunk_metadata,
        )

    @staticmethod
    def _decode_metadata(metadata: dict[str, Any] | None) -> dict[str, Any]:
        raw_metadata = (metadata or {}).get("chunk_metadata", "{}")
        try:
            decoded = json.loads(raw_metadata)
        except (TypeError, json.JSONDecodeError):
            return {}
        return decoded if isinstance(decoded, dict) else {}

    async def delete_document(self, document_id: str) -> None:
        await asyncio.to_thread(
            self._collection.delete,
            where={"document_id": document_id},
        )
