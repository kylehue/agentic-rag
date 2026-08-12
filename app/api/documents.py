from fastapi import APIRouter, File, Query, UploadFile

from app.dependencies import (
    document_service,
    embedder,
    ingestion_service,
    vector_storage,
)
from app.models.document import Document, DocumentChunk

router = APIRouter(prefix="/documents", tags=["Documents"])


@router.post("/", response_model=Document)
async def ingest_document(file: UploadFile = File(...)):
    return await ingestion_service.ingest(file)


@router.get("/", response_model=list[Document])
async def get_documents():
    return await document_service.list()


@router.get("/chunks/search", response_model=list[DocumentChunk])
async def search_document_chunks(
    query: str = Query(..., min_length=1, description="Natural-language search query"),
    top_k: int = Query(5, ge=1, le=100, description="Maximum chunks to return"),
):
    """Return the chunks most semantically relevant to a query."""
    query_embedding = await embedder.embed_query(query)
    return await vector_storage.search(query_embedding, top_k=top_k)


@router.get("/{document_id}", response_model=Document)
async def get_document(document_id: str):
    return await document_service.get(document_id)


@router.delete("/{document_id}", response_model=Document)
async def delete_document(document_id: str):
    return await document_service.delete(document_id)
