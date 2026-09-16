from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from fastapi.responses import StreamingResponse

from app.api.auth import require_user
from app.api.sse import sse_frame
from app.api_schemas.chunk import RetrievedChunkSchema
from app.api_schemas.rag import (
    DeleteFileResponseSchema,
    FileMetadataSchema,
    FileSchema,
    IngestJobSchema,
    ListFileChunksResponseSchema,
    ListFilesResponseSchema,
    RetrieveResponseSchema,
    StoredChunkSchema,
)
from app.container import chat_service, rag_service
from app.plugin.context import IngestionFile

router = APIRouter(prefix="/rag", tags=["RAG"])


@router.post(
    "/ingest",
    response_model=IngestJobSchema,
)
async def ingest_documents(
    files: list[UploadFile] | None = File(None),
    file: UploadFile | None = File(None),
    chat_id: str | None = Query(None),
    user: str = Depends(require_user),
):
    """Queue one or more files for background ingestion and return the job.

    Send multiple files as repeated `files` parts (a single legacy `file`
    part is also accepted). The response carries the `job_id`; watch progress
    with `GET /rag/ingest/stream?job_id=...`. The chat is created if not
    given; ingesting stamps the chat id onto the files and their chunks.
    """
    uploads = (files or []) + ([file] if file is not None else [])
    if not uploads:
        raise HTTPException(status_code=400, detail="No files provided.")
    chat_id = await chat_service.get_or_create(user, chat_id)
    ingestion_files = [
        IngestionFile(
            filename=upload.filename or "",
            content_type=upload.content_type or "",
            file_bytes=await upload.read(),
        )
        for upload in uploads
    ]
    job = rag_service.enqueue_ingest(ingestion_files, chat_id)
    return IngestJobSchema(
        chat_id=chat_id,
        job_id=job.job_id,
        status=job.status,
        files=[f.filename for f in ingestion_files],
    )


@router.get("/ingest/stream")
async def ingest_stream(job_id: str, user: str = Depends(require_user)):
    """An ingest job's progress as a server-sent event stream.

    Frames: a `chat` frame naming the chat, a `queued` frame per file, then
    per file a `started` frame, `stage` frames (processing / saving /
    embedding), `plugin_state` frames (the plugins' fine-grained states), and
    a `file_done` frame; the stream ends with a `done` frame (or an `error`
    frame). Late subscribers replay the frames already emitted.
    """
    job = rag_service.get_ingest_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Ingest job not found.")
    owner = await chat_service.username_of(job.chat_id)
    if owner != user:
        raise HTTPException(status_code=403, detail="Not your ingest job.")

    async def generate():
        async for event in job.events.subscribe():
            yield sse_frame(event.name, event.payload)

    return StreamingResponse(generate(), media_type="text/event-stream")


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
    "/files/{source_id}",
    response_model=FileMetadataSchema,
)
async def get_file(
    source_id: str,
    user: str = Depends(require_user),
):
    # The file's metadata and the link to retrieve it, for a file the user
    # owns. 404 for a missing file or one that belongs to someone else.
    document = await rag_service.get_file_metadata(source_id)
    if document is None:
        raise HTTPException(status_code=404, detail="File not found.")
    owner = await chat_service.username_of(document["chat_id"])
    if owner != user:
        raise HTTPException(status_code=404, detail="File not found.")
    link = await rag_service.get_file_link(source_id)
    if link is None:
        raise HTTPException(status_code=404, detail="File not found.")
    return FileMetadataSchema(
        source_id=source_id,
        filename=document["file_orig_filename"],
        content_type=document["file_content_type"],
        is_origin=bool(document["is_origin"]),
        chat_id=document["chat_id"],
        link=link,
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
