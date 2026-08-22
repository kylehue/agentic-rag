from fastapi import APIRouter, File, UploadFile

from app.container import rag_service
from app.models.chunk import IngestedChunk, RetrievedChunk
from app.services.rag import RagAnswer

router = APIRouter(prefix="/rag", tags=["RAG"])


@router.post("/ingest")
async def ingest_document(file: UploadFile = File(...)):
    return await rag_service.ingest(file)


@router.post("/retrieve")
async def retrieve(user_query: str):
    return await rag_service.retrieve(user_query)


@router.post("/answer")
async def answer(user_query: str):
    return await rag_service.answer(user_query)
