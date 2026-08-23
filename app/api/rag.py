from fastapi import APIRouter, File, UploadFile

from app.container import rag_service
from app.api_schemas.chunk import (
    IngestedChunkSchema,
    RetrievedChunkSchema,
)
from app.api_schemas.rag import RagAnswerSchema

router = APIRouter(prefix="/rag", tags=["RAG"])


@router.post("/ingest", response_model=list[IngestedChunkSchema])
async def ingest_document(file: UploadFile = File(...)):
    chunks = await rag_service.ingest(file)
    return [IngestedChunkSchema.model_validate(chunk) for chunk in chunks]


@router.post("/retrieve", response_model=list[RetrievedChunkSchema])
async def retrieve(user_query: str):
    chunks = await rag_service.retrieve(user_query)
    return [RetrievedChunkSchema.model_validate(chunk) for chunk in chunks]


@router.post("/answer", response_model=RagAnswerSchema)
async def answer(user_query: str):
    result = await rag_service.answer(user_query)
    return RagAnswerSchema.model_validate(result)
