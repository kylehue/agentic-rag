from fastapi import APIRouter
from app.container import document_service
from app.models.document import DocumentChunk

router = APIRouter(prefix="/documents", tags=["Documents"])


@router.get("/", response_model=list[DocumentChunk])
async def get_documents():
    return await document_service.get_all()


@router.get("/{document_id}", response_model=DocumentChunk)
async def get_document(document_id: str):
    return await document_service.get(document_id)


@router.delete("/{document_id}", response_model=DocumentChunk)
async def delete_document(document_id: str):
    return await document_service.delete(document_id)
