from fastapi import APIRouter, UploadFile, File

from app.services.document import DocumentService
from app.models.document import (
    Document,
    Document,
    DocumentUploadResponse,
    DocumentDeleteResponse,
)

router = APIRouter(prefix="/documents", tags=["Documents"])

document_service = DocumentService()


@router.post("/", response_model=DocumentUploadResponse)
async def upload_document(file: UploadFile = File(...)):
    return await document_service.upload(file)


@router.get("/", response_model=list[Document])
async def get_documents():
    return await document_service.list()


@router.get("/{document_id}", response_model=Document)
async def get_document(document_id: str):
    return await document_service.get(document_id)


@router.delete("/{document_id}", response_modeol=DocumentDeleteResponse)
async def delete_document(document_id: str):
    return await document_service.delete(document_id)
