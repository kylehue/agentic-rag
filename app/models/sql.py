from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SqlTableData:
    """A logical spreadsheet sheet before a backend assigns SQL identifiers."""

    source_name: str
    columns: list[str]
    rows: list[tuple[Any, ...]]


@dataclass(frozen=True)
class SqlColumn:
    source_name: str
    name: str
    type: str


@dataclass(frozen=True)
class StoredSqlTable:
    document_id: str
    source_name: str
    name: str
    columns: list[SqlColumn]


@dataclass(frozen=True)
class SqlQueryResult:
    columns: list[str]
    rows: list[tuple[Any, ...]]
