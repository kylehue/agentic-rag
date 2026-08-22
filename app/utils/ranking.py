from collections.abc import Callable, Sequence
from typing import TypeVar

T = TypeVar("T")


def reciprocal_rank_fusion(
    results: Sequence[Sequence[T]],
    *,
    id_fn: Callable[[T], str],
    k: int = 60,
) -> list[tuple[T, float]]:
    """
    Merge ranked result lists using Reciprocal Rank Fusion (RRF).

    Each input sequence must already be ordered from best to worst.

    RRF score:
        sum(1 / (k + rank))

    Rank is 1-based. Higher scores are better.
    """

    scores: dict[str, float] = {}
    items: dict[str, T] = {}

    for result in results:
        for rank, item in enumerate(result, start=1):
            item_id = id_fn(item)

            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + rank)

            # Keep the first instance of the item.
            items.setdefault(item_id, item)

    ranked_ids = sorted(
        scores,
        key=scores.__getitem__,
        reverse=True,
    )

    return [(items[item_id], scores[item_id]) for item_id in ranked_ids]
