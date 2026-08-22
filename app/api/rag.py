from fastapi import APIRouter, File, UploadFile

from app.container import rag_service
from app.models.document import Document
from app.models.rag import RagAnswer, RetrievalRequest, RetrievalResult

router = APIRouter(prefix="/rag", tags=["RAG"])


@router.post("/ingest", response_model=Document)
async def ingest_document(file: UploadFile = File(...)):
    return await rag_service.ingest(file)


@router.post("/retrieve", response_model=RetrievalResult)
async def retrieve(request: RetrievalRequest):
    return await rag_service.retrieve(request)


@router.post("/answer", response_model=RagAnswer)
async def answer(request: RetrievalRequest):
    return await rag_service.answer(request)
