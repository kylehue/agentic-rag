from typing import Any

from sqlalchemy import ColumnElement, Table

from app.database import CHUNK_TABLE_NAME
from app.models.chunk import RetrievedTextChunk
from app.retrievers.base import Retriever
from app.store_sql.base import ConditionBuilder, SqlStorage


class SparseRetriever(Retriever):
    def __init__(
        self,
        sql_storage: SqlStorage,
        top_k: int,
    ):
        self._sql_storage = sql_storage
        self._top_k = top_k

    async def retrieve(self, user_query, where: dict[str, Any] | None = None):
        raw_chunks = await self._sql_storage.search(
            CHUNK_TABLE_NAME,
            user_query,
            self._top_k,
            self._condition(where),
        )

        result = []
        for i, raw_chunk in enumerate(raw_chunks, 1):
            result.append(
                RetrievedTextChunk.from_dict(
                    raw_chunk,
                    score=0,  # doesn't matter for sparse search (as long as it's sorted)
                    text=raw_chunk["text"],
                )
            )

        return result

    @staticmethod
    def _condition(where: dict[str, Any] | None) -> ConditionBuilder | None:
        """The simple `{column: value}` map as a SQLAlchemy condition
        builder (AND of equalities, column order as given)."""
        if not where:
            return None

        def condition(table: Table) -> ColumnElement[bool]:
            expr: ColumnElement[bool] | None = None
            for column, value in where.items():
                clause = table.c[column] == value
                expr = clause if expr is None else expr & clause
            if expr is None:
                raise ValueError("Cannot build a condition from an empty map.")
            return expr

        return condition
