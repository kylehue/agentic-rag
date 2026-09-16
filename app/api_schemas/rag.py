from pydantic import BaseModel, ConfigDict

from app.api_schemas.chunk import RetrievedChunkSchema


class RagAnswerRequestSchema(BaseModel):
    """The question, and the chat it belongs to.

    A missing `chat_id` creates a new chat; the prior conversation lives in
    the chat, not in the request.
    """

    query: str
    chat_id: str | None = None


class RagAnswerSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    query: str
    answer: str
    # Only the chunks the answer actually cites, keyed by their
    # #[origin_source_id:chunk_id] reference.
    chunk_refs: dict[str, dict[str, str]]


class ChatAnswerSchema(RagAnswerSchema):
    # The chat the answer ran on (created if the request had none).
    chat_id: str


class RagStopRequestSchema(BaseModel):
    """The chat whose in-flight answer should be interrupted. Unlike the
    answer request, a missing chat is an error (there is nothing to create)."""

    chat_id: str


class RagStopResponseSchema(BaseModel):
    chat_id: str
    # True when a run was in flight and got interrupted; False when the chat
    # had no run to stop.
    stopped: bool


class IngestJobSchema(BaseModel):
    """The acknowledgement for a queued (background) ingestion: the job id
    addresses the progress stream, not the (not-yet-done) chunks."""

    chat_id: str
    job_id: str
    status: str
    files: list[str]


class RetrieveResponseSchema(BaseModel):
    chat_id: str
    chunks: list[RetrievedChunkSchema]


class FileSchema(BaseModel):
    """An origin file the user ingested (not a plugin-emitted file)."""

    source_id: str
    filename: str
    content_type: str
    chat_id: str | None


class FileMetadataSchema(BaseModel):
    """The metadata of one stored file, with the link to retrieve it."""

    source_id: str
    filename: str
    content_type: str
    is_origin: bool
    chat_id: str | None
    link: str


class ListFilesResponseSchema(BaseModel):
    chat_id: str
    files: list[FileSchema]


class StoredChunkSchema(BaseModel):
    """A stored chunk row (from the SQL store, not a retrieval result)."""

    chunk_id: str
    source_id: str
    parent_source_id: str | None
    origin_source_id: str
    plugin: str
    text: str
    metadata: dict
    chat_id: str | None


class ListFileChunksResponseSchema(BaseModel):
    chat_id: str
    origin_source_id: str
    chunks: list[StoredChunkSchema]


class DeleteFileResponseSchema(BaseModel):
    chat_id: str
    origin_source_id: str
    deleted_files: int
