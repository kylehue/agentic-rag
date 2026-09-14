from pydantic import BaseModel, ConfigDict

from app.api_schemas.chunk import IngestedChunkSchema, RetrievedChunkSchema


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
    # #[source_id:chunk_id] reference.
    chunk_refs: dict[str, dict[str, str]]


class ChatAnswerSchema(RagAnswerSchema):
    # The chat the answer ran on (created if the request had none).
    chat_id: str


class IngestResponseSchema(BaseModel):
    chat_id: str
    chunks: list[IngestedChunkSchema]


class RetrieveResponseSchema(BaseModel):
    chat_id: str
    chunks: list[RetrievedChunkSchema]
