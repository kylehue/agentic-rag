from collections.abc import Sequence

from app.finalizers.base import Finalizer


class HybridFinalizer(Finalizer):
    def __init__(self, finalizers: Sequence[Finalizer]):
        self._finalizers = finalizers

    async def finalize(self, user_query, chunk):
        for finalizer in self._finalizers:
            chunk = await finalizer.finalize(user_query, chunk)

        return chunk
