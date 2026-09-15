from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile

from app.api.auth import require_user
from app.api_schemas.chunk import IngestedChunkSchema, RetrievedChunkSchema
from app.api_schemas.rag import (
    DeleteFileResponseSchema,
    FileSchema,
    IngestResponseSchema,
    ListFileChunksResponseSchema,
    ListFilesResponseSchema,
    RetrieveResponseSchema,
    StoredChunkSchema,
)
from app.container import chat_service, rag_service

router = APIRouter(prefix="/rag", tags=["RAG"])


@router.post(
    "/ingest",
    response_model=IngestResponseSchema,
)
async def ingest_document(
    file: UploadFile = File(...),
    chat_id: str | None = Query(None),
    user: str = Depends(require_user),
):
    # The HTTP boundary is the only place that touches UploadFile; the RAG
    # core ingests plain bytes. The chat is created if not given; ingesting
    # stamps the chat id onto the file and its chunks, so nothing else needs
    # to record it.
    chat_id = await chat_service.get_or_create(user, chat_id)
    _origin_source_id, chunks = await rag_service.ingest(
        file_bytes=await file.read(),
        filename=file.filename or "",
        content_type=file.content_type or "",
        chat_id=chat_id,
    )
    return IngestResponseSchema(
        chat_id=chat_id,
        chunks=[IngestedChunkSchema.model_validate(chunk) for chunk in chunks],
    )


@router.post(
    "/retrieve",
    response_model=RetrieveResponseSchema,
)
async def retrieve(
    user_query: str,
    chat_id: str | None = Query(None),
    user: str = Depends(require_user),
):
    # Retrieval is bounded to the chat's chunks (applied at the index).
    chat_id = await chat_service.get_or_create(user, chat_id)
    chunks = await rag_service.retrieve(user_query, chat_id)
    return RetrieveResponseSchema(
        chat_id=chat_id,
        chunks=[RetrievedChunkSchema.model_validate(chunk) for chunk in chunks],
    )


@router.get(
    "/files",
    response_model=ListFilesResponseSchema,
)
async def list_files(
    chat_id: str | None = Query(None),
    user: str = Depends(require_user),
):
    # Only origin files (user uploads), scoped to the chat.
    chat_id = await chat_service.get_or_create(user, chat_id)
    documents = await rag_service.list_files(chat_id)
    return ListFilesResponseSchema(
        chat_id=chat_id,
        files=[
            FileSchema(
                source_id=doc["source_id"],
                filename=doc["file_orig_filename"],
                content_type=doc["file_content_type"],
                chat_id=doc["chat_id"],
            )
            for doc in documents
        ],
    )


@router.get(
    "/files/{origin_source_id}/chunks",
    response_model=ListFileChunksResponseSchema,
)
async def list_file_chunks(
    origin_source_id: str,
    chat_id: str | None = Query(None),
    user: str = Depends(require_user),
):
    # All chunks of the file's emission tree (origin + emitted descendants),
    # bounded to the chat.
    chat_id = await chat_service.get_or_create(user, chat_id)
    rows = await rag_service.list_file_chunks(origin_source_id, chat_id)
    return ListFileChunksResponseSchema(
        chat_id=chat_id,
        origin_source_id=origin_source_id,
        chunks=[StoredChunkSchema.model_validate(row) for row in rows],
    )


@router.delete(
    "/files/{origin_source_id}",
    response_model=DeleteFileResponseSchema,
)
async def delete_file(
    origin_source_id: str,
    chat_id: str | None = Query(None),
    user: str = Depends(require_user),
):
    # Reverse ingestion for the file's whole emission tree, bounded to the
    # chat (a file not in this chat is a 404, not a silent no-op).
    chat_id = await chat_service.get_or_create(user, chat_id)
    deleted = await rag_service.delete_file(origin_source_id, chat_id)
    if deleted == 0:
        raise HTTPException(status_code=404, detail="File not found in this chat.")
    return DeleteFileResponseSchema(
        chat_id=chat_id,
        origin_source_id=origin_source_id,
        deleted_files=deleted,
    )
