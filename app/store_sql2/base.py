from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any, ClassVar
from sqlalchemy import Column


class SqlStorage(ABC):
    @abstractmethod
    def get_sql_dialect(self) -> str:
        """Returns the SQL dialect used for queries. Used for LLM SQL query generation."""

    @abstractmethod
    async def close(self) -> None:
        """Close the database connection pool."""

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
        conflict_columns: Sequence[str],
    ) -> None:
        """Insert (or update) rows to a database table."""

    @abstractmethod
    async def get(
        self,
        table_name: str,
        row_id: str,
    ) -> dict[str, Any] | None:
        """Get a particular row in a database table."""

    @abstractmethod
    async def get_all(
        self,
        table_name: str,
    ) -> Sequence[dict[str, Any]]:
        """Get all rows in a database table."""

    @abstractmethod
    async def delete(
        self,
        table_name: str,
        row_id: str,
    ) -> bool:
        """Deletes a particular row in a database table."""

    @abstractmethod
    async def query(
        self,
        sql_query: str,
        limit: int,
    ) -> Sequence[dict[str, Any]]:
        """Read-only query to a database table. Returns row results."""

    @abstractmethod
    async def search(
        self,
        table_name: str,
        search_query: str,
        limit: int,
    ) -> Sequence[dict[str, Any]]:
        """Search a database table using FTS5. Returns row results sorted by most relevant to least relevant."""

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
