from typing import Any

from pydantic import BaseModel, Field

from app.models.document import Document, DocumentCategory


class RetrievalRequest(BaseModel):
    """The input for retrieval and answer synthesis."""

    query: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=100)
    categories: set[DocumentCategory] | None = None
    max_sql_rows: int = Field(default=100, ge=1, le=1_000)


class RetrievalCandidate(BaseModel):
    """A raw result from candidate generation, before source interpretation."""

    id: str
    text: str
    document: Document
    metadata: dict[str, Any] = Field(default_factory=dict)
    score: float | None = None
    binary_content: bytes | None = Field(default=None, exclude=True)
    binary_mime_type: str | None = Field(default=None, exclude=True)


class RetrievedEvidence(BaseModel):
    """The one output contract that every retriever returns."""

    id: str
    retriever: str
    document: Document
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    score: float | None = None
    binary_content: bytes | None = Field(default=None, exclude=True)
    binary_mime_type: str | None = Field(default=None, exclude=True)


class RetrievalResult(BaseModel):
    query: str
    evidence: list[RetrievedEvidence]


class RagAnswer(BaseModel):
    query: str
    answer: str
    evidence: list[RetrievedEvidence]
