from fastapi import APIRouter, UploadFile, File

from app.dependencies import document_service
from app.models.document import Document

router = APIRouter(prefix="/documents", tags=["Documents"])


@router.post("/", response_model=Document)
async def upload_document(file: UploadFile = File(...)):
    return await document_service.upload(file)


@router.get("/", response_model=list[Document])
async def get_documents():
    return await document_service.list()


@router.get("/{document_id}", response_model=Document)
async def get_document(document_id: str):
    return await document_service.get(document_id)


@router.delete("/{document_id}", response_model=Document)
async def delete_document(document_id: str):
    return await document_service.delete(document_id)
