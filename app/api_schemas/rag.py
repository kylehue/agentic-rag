from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict

from app.api_schemas.chunk import RetrievedChunkSchema


class RagAnswerSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    query: str
    answer: str
    chunks: Sequence[RetrievedChunkSchema]
