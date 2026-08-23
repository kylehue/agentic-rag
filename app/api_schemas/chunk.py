from pydantic import BaseModel, ConfigDict

from app.models.chunk import ChunkCategory


class IngestedChunkSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    chunk_id: str
    category: ChunkCategory
    text: str
    metadata: dict


class RetrievedChunkSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    chunk_id: str
    source_id: str
    category: ChunkCategory
    text: str
    score: float
    metadata: dict
