import re

# The citation marker the model writes in its answer: #[1], #[2], ...
CITE_PATTERN = re.compile(r"#\[(\d+)\]")


class EvidenceIndex:
    """A per-answer map from a small integer to a real chunk.

    Chunk-surfacing tools register the chunks they return and get back an
    integer to show the model (``[1]``, ``[2]``, ...); the service later
    resolves the integers the model actually cited back to real ids. The same
    chunk (by origin + chunk id) always maps to the same integer, so a chunk
    the model sees twice carries one number.
    """

    def __init__(self) -> None:
        self._by_chunk: dict[tuple[str, str], int] = {}
        self._by_index: dict[int, tuple[str, str]] = {}

    def register(self, origin_source_id: str, chunk_id: str) -> int:
        key = (origin_source_id, chunk_id)
        if key in self._by_chunk:
            return self._by_chunk[key]
        index = len(self._by_index) + 1
        self._by_chunk[key] = index
        self._by_index[index] = key
        return index

    def get(self, index: int) -> tuple[str, str] | None:
        """The (origin_source_id, chunk_id) behind an index, or None."""
        return self._by_index.get(index)


def resolve_citations(
    answer: str, evidence: EvidenceIndex
) -> dict[str, dict[str, str]]:
    """The evidence the answer actually cited.

    Scans the answer for ``#[n]`` markers and resolves each to the real
    ``{chunk_id, origin_source_id}`` the evidence index holds. A marker that
    was never surfaced is dropped, and chunks that were surfaced but not cited
    are omitted, so the result is exactly the evidence the answer relies on.
    Keyed by the marker (``"#[1]"``) so a client can map answer text straight
    to ids.
    """
    refs: dict[str, dict[str, str]] = {}
    for match in CITE_PATTERN.finditer(answer):
        hit = evidence.get(int(match.group(1)))
        if hit is None:
            continue
        origin_source_id, chunk_id = hit
        refs.setdefault(
            match.group(0),
            {"chunk_id": chunk_id, "origin_source_id": origin_source_id},
        )
    return refs
