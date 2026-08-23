from app.models.chunk import RetrievedChunk


from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class RagAnswer:
    query: str
    answer: str
    chunks: Sequence[RetrievedChunk]
