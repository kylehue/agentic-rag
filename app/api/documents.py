from fastapi import APIRouter
from app.dependencies import document_service
from app.models.document import Document

router = APIRouter(prefix="/documents", tags=["Documents"])


@router.get("/", response_model=list[Document])
async def get_documents():
    return await document_service.get_all()


@router.get("/{document_id}", response_model=Document)
async def get_document(document_id: str):
    return await document_service.get(document_id)


@router.delete("/{document_id}", response_model=Document)
async def delete_document(document_id: str):
    return await document_service.delete(document_id)
