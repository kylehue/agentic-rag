from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict

from app.models.chunk import RetrievedChunk


class RagAnswerSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    query: str
    answer: str
    chunks: Sequence[RetrievedChunk]
