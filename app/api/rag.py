from fastapi import APIRouter, Depends, File, Query, UploadFile

from app.api.auth import require_user
from app.api_schemas.chunk import IngestedChunkSchema, RetrievedChunkSchema
from app.api_schemas.rag import IngestResponseSchema, RetrieveResponseSchema
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
