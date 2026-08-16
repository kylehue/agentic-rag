from abc import ABC, abstractmethod
from collections.abc import Sequence

from app.models.sql import SqlQueryResult, SqlTableData, StoredSqlTable


class SqlStorage(ABC):
    @abstractmethod
    async def replace_document_tables(
        self,
        document_id: str,
        tables: Sequence[SqlTableData],
    ) -> list[StoredSqlTable]:
        """Replace every SQL table owned by a source document."""

    @abstractmethod
    async def list_document_tables(self, document_id: str) -> list[StoredSqlTable]:
        """Return table/column mappings for an LLM query planner."""

    @abstractmethod
    async def query(
        self, document_id: str, sql: str, max_rows: int = 100
    ) -> SqlQueryResult:
        """Execute a read-only query scoped to one document's tables."""

    @abstractmethod
    async def delete_document(self, document_id: str) -> None:
        """Delete all structured data owned by a document."""
