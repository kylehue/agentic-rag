from pydantic import BaseModel, ConfigDict


class IngestedChunkSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    plugin: str
    chunk_id: str
    text: str
    metadata: dict


class RetrievedChunkSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    plugin: str
    chunk_id: str
    source_id: str
    origin_source_id: str
    parent_source_id: str | None
    text: str
    score: float
    metadata: dict
