from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any, ClassVar
from sqlalchemy import Column, ColumnElement, Table


class SqlStorage(ABC):
    @abstractmethod
    def get_sql_dialect(self) -> str:
        """Returns the SQL dialect used for queries. Used for LLM SQL query generation."""

    @abstractmethod
    async def close(self) -> None:
        """Close the database connection pool."""

    @abstractmethod
    async def get_table(self, table_name: str) -> Table:
        """Load the existing table in SQL Alchemy format."""

    @abstractmethod
    async def ensure_table(
        self,
        table_name: str,
        columns: Sequence[Column],
    ) -> None:
        """Creates a table if it doesn't exist."""

    @abstractmethod
    async def upsert(
        self,
        table_name: str,
        rows: Sequence[dict[str, Any]],
        conflict_columns: Sequence[str] = (),
    ) -> None:
        """Insert (or update) rows to a database table."""

    @abstractmethod
    async def get(
        self,
        table_name: str,
        conditions: Sequence[ColumnElement[bool]],
    ) -> dict[str, Any] | None:
        """Get the first row matching all conditions."""

    @abstractmethod
    async def get_all(
        self,
        table_name: str,
        conditions: Sequence[ColumnElement[bool]] = (),
        limit: int | None = None,
    ) -> Sequence[dict[str, Any]]:
        """Get rows matching all conditions."""

    @abstractmethod
    async def delete(
        self,
        table_name: str,
        conditions: Sequence[ColumnElement[bool]],
    ) -> bool:
        """Delete rows matching all conditions."""

    @abstractmethod
    async def query(
        self,
        sql_query: str,
        limit: int | None,
    ) -> Sequence[dict[str, Any]]:
        """Read-only query to a database table. Returns row results."""

    @abstractmethod
    async def search(
        self,
        table_name: str,
        search_query: str,
        limit: int | None,
    ) -> Sequence[dict[str, Any]]:
        """
        Search a database table using FTS5
        Returns row results sorted by most relevant to least relevant.
        """

    @staticmethod
    @abstractmethod
    def create_sql_columns_from_schema(
        schema: list[dict[str, Any]],
    ) -> list[Column[Any]]:
        """
        Create SQLAlchemy columns from the schema.

        The schema is expected to be in form of:
        ```
        [
            {
                name: "column name",
                type: "INTEGER" | "BOOLEAN" | "REAL" | "TEXT" | "DATETIME"
            }
            ...
        ]
        ```
        """
