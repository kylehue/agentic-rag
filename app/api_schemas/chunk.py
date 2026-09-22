from pydantic import BaseModel, ConfigDict


class IngestedChunkSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    plugin: str
    chunk_id: str
    # None for image chunks (their content is the image, not text).
    text: str | None = None
    metadata: dict


class RetrievedChunkSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    plugin: str
    chunk_id: str
    source_id: str
    origin_source_id: str
    parent_source_id: str | None
    # None for image chunks (their content is the image, fetched via source_id).
    text: str | None = None
    score: float
    metadata: dict
