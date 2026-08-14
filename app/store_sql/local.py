import asyncio
import json
import re
import sqlite3
from contextlib import contextmanager
from collections.abc import Sequence
from datetime import date, datetime, time
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Iterator

from app.core.config import settings
from app.store_sql.base import (
    SqlColumn,
    SqlQueryResult,
    SqlStorage,
    SqlTableData,
    StoredSqlTable,
)


class LocalSqlStorage(SqlStorage):
    """SQLite implementation. Other SQL backends only need to implement SqlStorage."""

    def __init__(self, database_path: str | Path | None = None):
        self._path = Path(database_path or settings.SQL_LOCAL_STORAGE_DIR)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self._path)

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._connection() as connection:
            connection.execute("""CREATE TABLE IF NOT EXISTS rag_sql_table_catalog (
                document_id TEXT NOT NULL, source_name TEXT NOT NULL,
                table_name TEXT PRIMARY KEY, columns_json TEXT NOT NULL)""")
            connection.execute(
                """CREATE INDEX IF NOT EXISTS idx_rag_sql_table_catalog_document
                ON rag_sql_table_catalog(document_id)"""
            )

    async def replace_document_tables(
        self,
        document_id: str,
        tables: Sequence[SqlTableData],
    ) -> list[StoredSqlTable]:
        return await asyncio.to_thread(
            self._replace_document_tables, document_id, list(tables)
        )

    def _replace_document_tables(
        self,
        document_id: str,
        tables: list[SqlTableData],
    ) -> list[StoredSqlTable]:
        with self._connection() as connection:
            self._delete_document(connection, document_id)
            stored: list[StoredSqlTable] = []
            for i, table in enumerate(tables):
                if not table.columns:
                    continue
                name = f"spreadsheet_{self._identifier(document_id)}_{i}"
                columns = self._columns(table)
                definition = ", ".join(
                    f"{self._quote(column.name)} {column.type}" for column in columns
                )
                connection.execute(f"CREATE TABLE {self._quote(name)} ({definition})")
                if table.rows:
                    placeholders = ", ".join("?" for _ in columns)
                    rows = [
                        tuple(
                            self._normalize(row[i]) if i < len(row) else None
                            for i in range(len(columns))
                        )
                        for row in table.rows
                    ]
                    connection.executemany(
                        f"INSERT INTO {self._quote(name)} VALUES ({placeholders})", rows
                    )
                descriptor = StoredSqlTable(
                    document_id, table.source_name, name, columns
                )
                connection.execute(
                    "INSERT INTO rag_sql_table_catalog VALUES (?, ?, ?, ?)",
                    (
                        document_id,
                        table.source_name,
                        name,
                        json.dumps([column.__dict__ for column in columns]),
                    ),
                )
                stored.append(descriptor)
        return stored

    async def list_document_tables(self, document_id: str) -> list[StoredSqlTable]:
        return await asyncio.to_thread(self._list_document_tables, document_id)

    def _list_document_tables(self, document_id: str) -> list[StoredSqlTable]:
        with self._connection() as connection:
            rows = connection.execute(
                """SELECT source_name, table_name, columns_json
                FROM rag_sql_table_catalog WHERE document_id = ? ORDER BY table_name""",
                (document_id,),
            ).fetchall()
        return [
            StoredSqlTable(
                document_id,
                source,
                table,
                [SqlColumn(**item) for item in json.loads(columns)],
            )
            for source, table, columns in rows
        ]

    async def query(
        self, document_id: str, sql: str, max_rows: int = 100
    ) -> SqlQueryResult:
        if not 1 <= max_rows <= 1_000:
            raise ValueError("max_rows must be between 1 and 1000")
        return await asyncio.to_thread(self._query, document_id, sql, max_rows)

    def _query(self, document_id: str, sql: str, max_rows: int) -> SqlQueryResult:
        statement = self._read_only_statement(sql)
        with self._connection() as connection:
            allowed = {
                row[0]
                for row in connection.execute(
                    "SELECT table_name FROM rag_sql_table_catalog WHERE document_id = ?",
                    (document_id,),
                )
            }
            connection.set_authorizer(self._authorizer(allowed))
            try:
                cursor = connection.execute(
                    f"SELECT * FROM ({statement}) AS rag_query_result LIMIT ?",
                    (max_rows,),
                )
                columns = [item[0] for item in cursor.description or []]
                rows = [
                    tuple(self._normalize(value) for value in row) for row in cursor
                ]
            finally:
                connection.set_authorizer(None)
        return SqlQueryResult(columns, rows)

    async def delete_document(self, document_id: str) -> None:
        await asyncio.to_thread(self._delete_document_by_id, document_id)

    def _delete_document_by_id(self, document_id: str) -> None:
        with self._connection() as connection:
            self._delete_document(connection, document_id)

    @classmethod
    def _delete_document(cls, connection: sqlite3.Connection, document_id: str) -> None:
        tables = connection.execute(
            "SELECT table_name FROM rag_sql_table_catalog WHERE document_id = ?",
            (document_id,),
        ).fetchall()
        for (table_name,) in tables:
            connection.execute(f"DROP TABLE IF EXISTS {cls._quote(table_name)}")
        connection.execute(
            "DELETE FROM rag_sql_table_catalog WHERE document_id = ?", (document_id,)
        )

    @classmethod
    def _columns(cls, table: SqlTableData) -> list[SqlColumn]:
        used: set[str] = set()
        result = []
        for i, source in enumerate(table.columns):
            base = cls._identifier(source) or f"column_{i + 1}"
            name, suffix = base, 2
            while name in used:
                name, suffix = f"{base}_{suffix}", suffix + 1
            used.add(name)
            values = [row[i] for row in table.rows if len(row) > i]
            result.append(SqlColumn(str(source), name, cls._column_type(values)))
        return result

    @staticmethod
    def _read_only_statement(sql: str) -> str:
        statement = sql.strip()
        if statement.endswith(";"):
            statement = statement[:-1].rstrip()
        if not statement or ";" in statement:
            raise ValueError("SQL must contain exactly one statement")
        if not statement.casefold().startswith(("select", "with")):
            raise ValueError("Only SELECT or WITH queries are allowed")
        return statement

    @staticmethod
    def _authorizer(allowed_tables: set[str]):
        allowed_actions = {
            sqlite3.SQLITE_SELECT,
            sqlite3.SQLITE_FUNCTION,
            sqlite3.SQLITE_RECURSIVE,
        }

        def authorize(
            action: int,
            parameter_1: str | None,
            parameter_2: str | None,
            database: str | None,
            trigger: str | None,
        ) -> int:
            if action == sqlite3.SQLITE_READ:
                return (
                    sqlite3.SQLITE_OK
                    if parameter_1 in allowed_tables
                    else sqlite3.SQLITE_DENY
                )
            return (
                sqlite3.SQLITE_OK if action in allowed_actions else sqlite3.SQLITE_DENY
            )

        return authorize

    @classmethod
    def _column_type(cls, values: list[Any]) -> str:
        values = [cls._normalize(value) for value in values]
        values = [value for value in values if value is not None]
        if values and all(isinstance(value, int) for value in values):
            return "INTEGER"
        if values and all(isinstance(value, (int, float)) for value in values):
            return "REAL"
        return "TEXT"

    @staticmethod
    def _normalize(value: Any) -> Any:
        if value is None or (isinstance(value, float) and value != value):
            return None
        if isinstance(value, (datetime, date, time)):
            return value.isoformat()
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, Integral):
            return int(value)
        if isinstance(value, Real):
            return float(value)
        if hasattr(value, "item"):
            return LocalSqlStorage._normalize(value.item())  # type: ignore
        return str(value)

    @staticmethod
    def _identifier(value: Any) -> str:
        return re.sub(r"[^a-zA-Z0-9_]", "_", str(value)).strip("_").lower()[:50]

    @staticmethod
    def _quote(identifier: str) -> str:
        return '"' + identifier.replace('"', '""') + '"'
