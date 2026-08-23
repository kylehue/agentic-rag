from dataclasses import dataclass


@dataclass(frozen=True)
class VectorSearchResult:
    id: str
    score: float
