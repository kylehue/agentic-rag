from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any
from sqlalchemy import Column


class SqlStorage(ABC):
    @abstractmethod
    async def ensure_table(
        self,
        db_name: str,
        table_name: str,
        columns: Sequence[Column],
    ) -> None:
        pass

    @abstractmethod
    async def upsert(
        self,
        db_name: str,
        table_name: str,
        rows: Sequence[dict[str, Any]],
        conflict_columns: Sequence[str],
    ) -> None:
        """Insert (or update) rows to a database table."""

    @abstractmethod
    async def get(
        self,
        db_name: str,
        table_name: str,
        row_id: str,
    ) -> dict[str, Any] | None:
        """Get a particular row in a database table."""

    @abstractmethod
    async def get_all(
        self,
        db_name: str,
        table_name: str,
    ) -> Sequence[dict[str, Any]]:
        """Get all rows in a database table."""

    @abstractmethod
    async def delete(
        self,
        db_name: str,
        table_name: str,
        row_id: str,
    ) -> bool:
        """Deletes a particular row in a database table."""

    @abstractmethod
    async def query(
        self,
        db_name: str,
        sql_query: str,
        limit: int,
    ) -> Sequence[dict[str, Any]]:
        """Read-only query to a database table. Returns row results."""

    @abstractmethod
    async def search(
        self,
        db_name: str,
        table_name: str,
        search_query: str,
        limit: int,
    ) -> Sequence[dict[str, Any]]:
        """Search a database table using FTS5. Returns row results sorted by most relevant to least relevant."""
