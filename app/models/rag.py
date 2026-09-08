from collections.abc import Sequence
from dataclasses import dataclass

from app.models.chunk import RetrievedChunk


@dataclass(frozen=True)
class RagAnswer:
    query: str
    answer: str
    chunks: Sequence[RetrievedChunk]
