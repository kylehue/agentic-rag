from fastapi import APIRouter, File, UploadFile

from app.api_schemas.chunk import (
    IngestedChunkSchema,
    RetrievedChunkSchema,
)
from app.container import rag_service

router = APIRouter(prefix="/rag", tags=["RAG"])


@router.post("/ingest", response_model=list[IngestedChunkSchema])
async def ingest_document(file: UploadFile = File(...)):
    file_bytes = await file.read()
    chunks = await rag_service.ingest(
        file_bytes=file_bytes,
        filename=file.filename or "",
        content_type=file.content_type or "",
    )
    return [IngestedChunkSchema.model_validate(chunk) for chunk in chunks]


@router.post("/retrieve", response_model=list[RetrievedChunkSchema])
async def retrieve(user_query: str):
    chunks = await rag_service.retrieve(user_query)
    return [RetrievedChunkSchema.model_validate(chunk) for chunk in chunks]
