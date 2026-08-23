from dataclasses import dataclass


@dataclass(frozen=True)
class VectorSearchResult:
    chunk_id: str
    score: float
