from fastapi import APIRouter, UploadFile, File

from app.dependencies import document_service
from app.models.document import Document, DocumentChunk
from unstructured.partition.auto import partition
from unstructured.partition.csv import partition_csv
from unstructured.partition.image import partition_image
from app.processors.pipeline import process

from unstructured.documents.elements import Element

router = APIRouter(prefix="/documents", tags=["Documents"])


@router.post("/", response_model=Document)
async def upload_document(file: UploadFile = File(...)):
    return await document_service.upload(file)


@router.post("/process")
async def process_document(file: UploadFile = File(...)):
    document = await document_service.upload(file)
    elements = partition(
        filename=document.path,
        strategy="hi_res",
        infer_table_structure=True,  # Keep tables as structured HTML, not jumbled text
        extract_image_block_types=["Image"],  # Grab images found in the PDF
        extract_image_block_to_payload=True,  # Store images as base64 data you can actually use
    )
    chunks = await process(document, elements)
    return [e.model_dump() for e in chunks]


@router.get("/", response_model=list[Document])
async def get_documents():
    return await document_service.list()


@router.get("/{document_id}", response_model=Document)
async def get_document(document_id: str):
    return await document_service.get(document_id)


@router.delete("/{document_id}", response_model=Document)
async def delete_document(document_id: str):
    return await document_service.delete(document_id)
