from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from sqlalchemy import (
    Column,
    ColumnElement,
    Integer,
    Float,
    String,
    Boolean,
    DateTime,
    MetaData,
    Table,
    delete,
    select,
    text,
)
from sqlalchemy.dialects.sqlite import insert
from sqlalchemy.exc import NoSuchTableError
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    async_sessionmaker,
    create_async_engine,
)

from app.store_sql.base import SqlStorage


class LocalSqlStorage(SqlStorage):
    """SQLite-backed SQL storage using one local database."""

    DB_FILENAME = "database.db"

    def __init__(
        self,
        storage_dir: str | Path,
    ) -> None:
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        database_path = self.storage_dir / self.DB_FILENAME

        self._engine: AsyncEngine = create_async_engine(
            f"sqlite+aiosqlite:///{database_path}",
            connect_args={"timeout": 30},
            pool_pre_ping=True,
        )

        self._session_factory = async_sessionmaker(
            self._engine,
            expire_on_commit=False,
        )

    @staticmethod
    def _validate_identifier(
        value: str,
        *,
        name: str,
    ) -> None:
        if not value:
            raise ValueError(f"{name} cannot be empty.")

        if len(value) > 255:
            raise ValueError(f"{name} is too long.")

        if not (value[0].isalpha() or value[0] == "_"):
            raise ValueError(f"Invalid {name}: {value!r}")

        if not all(char.isalnum() or char == "_" for char in value):
            raise ValueError(f"Invalid {name}: {value!r}")

    def _validate_table_name(self, table_name: str) -> None:
        self._validate_identifier(
            table_name,
            name="table name",
        )

    async def _load_table(self, table_name: str) -> Table:
        """Reflect an existing table from SQLite."""

        self._validate_table_name(table_name)

        metadata = MetaData()

        async with self._engine.connect() as conn:

            def load(sync_conn) -> Table:
                return Table(
                    table_name,
                    metadata,
                    autoload_with=sync_conn,
                )

            try:
                return await conn.run_sync(load)
            except NoSuchTableError as exc:
                raise ValueError(f"Table {table_name!r} does not exist.") from exc

    async def close(self) -> None:
        await self._engine.dispose()

    def get_sql_dialect(self) -> str:
        return "SQLite"

    async def get_table(self, table_name: str) -> Table:
        return await self._load_table(table_name)

    async def ensure_table(
        self,
        table_name: str,
        columns: Sequence[Column[Any]],
    ) -> None:
        self._validate_table_name(table_name)

        if not columns:
            raise ValueError("A table must contain at least one column.")

        metadata = MetaData()

        Table(
            table_name,
            metadata,
            *columns,
        )

        async with self._engine.begin() as conn:
            await conn.run_sync(metadata.create_all)

    async def upsert(
        self,
        table_name: str,
        rows: Sequence[dict[str, Any]],
        conflict_columns: Sequence[str] = (),
    ) -> None:
        if not rows:
            return

        table = await self._load_table(table_name)

        table_columns = {column.name for column in table.columns}

        for column in conflict_columns:
            if column not in table_columns:
                raise ValueError(f"Unknown conflict column: {column!r}")

        for row in rows:
            unknown = set(row) - table_columns

            if unknown:
                raise ValueError(f"Unknown columns: {sorted(unknown)}")

        statement = insert(table).values(list(rows))

        if conflict_columns:
            update_columns = {
                column.name: statement.excluded[column.name]
                for column in table.columns
                if column.name not in conflict_columns
            }

            if update_columns:
                statement = statement.on_conflict_do_update(
                    index_elements=list(conflict_columns),
                    set_=update_columns,
                )
            else:
                statement = statement.on_conflict_do_nothing(
                    index_elements=list(conflict_columns),
                )

        async with self._session_factory() as session:
            async with session.begin():
                await session.execute(statement)

    async def get(
        self,
        table_name: str,
        conditions: Sequence[ColumnElement[bool]],
    ) -> dict[str, Any] | None:
        if not conditions:
            raise ValueError("Get requires at least one condition.")

        table = await self.get_table(table_name)

        statement = select(table)

        if conditions:
            statement = statement.where(*conditions)

        statement = statement.limit(1)

        async with self._session_factory() as session:
            result = await session.execute(statement)

            row = result.mappings().first()

            if row is None:
                return None

            return dict(row)

    async def get_all(
        self,
        table_name: str,
        conditions: Sequence[ColumnElement[bool]] = (),
        limit: int | None = None,
    ) -> Sequence[dict[str, Any]]:
        table = await self.get_table(table_name)

        statement = select(table)

        if conditions:
            statement = statement.where(*conditions)

        if limit is not None:
            if limit <= 0:
                return []

            statement = statement.limit(limit)

        async with self._session_factory() as session:
            result = await session.execute(statement)

            return [dict(row) for row in result.mappings()]

    async def delete(
        self,
        table_name: str,
        conditions: Sequence[ColumnElement[bool]],
    ) -> bool:
        table = await self.get_table(table_name)

        if not conditions:
            raise ValueError("Delete requires at least one condition.")

        if not table.c:
            raise ValueError(f"Table {table_name!r} has no columns.")

        statement = delete(table)

        if conditions:
            statement = statement.where(*conditions)

        return_column = next(iter(table.c))

        statement = statement.returning(return_column)

        async with self._session_factory() as session:
            async with session.begin():
                result = await session.execute(statement)

                return result.first() is not None

    async def query(
        self,
        sql_query: str,
        limit: int | None,
    ) -> Sequence[dict[str, Any]]:
        query = sql_query.strip()

        if not query:
            raise ValueError("SQL query cannot be empty.")

        if not query.lower().startswith("select"):
            raise ValueError("Only SELECT statements are allowed.")

        if ";" in query.rstrip(";"):
            raise ValueError("Multiple SQL statements are not allowed.")

        statement = text(query)

        async with self._session_factory() as session:
            result = await session.execute(statement)

            if limit is None:
                rows = result.mappings().all()
            else:
                rows = result.mappings().fetchmany(limit)

            return [dict(row) for row in rows]

    async def search(
        self,
        table_name: str,
        search_query: str,
        limit: int | None,
    ) -> Sequence[dict[str, Any]]:
        self._validate_table_name(table_name)

        fts_table = f"{table_name}_fts"

        statement = text(f"""
            SELECT *
            FROM "{fts_table}"
            WHERE "{fts_table}" MATCH :query
            LIMIT :limit
            """)

        async with self._session_factory() as session:
            result = await session.execute(
                statement,
                {
                    "query": search_query,
                    "limit": limit,
                },
            )

            return [dict(row) for row in result.mappings()]

    @staticmethod
    def create_sql_columns_from_schema(
        schema: list[dict[str, Any]],
    ) -> list[Column[Any]]:
        type_map = {
            "INTEGER": Integer,
            "REAL": Float,
            "TEXT": String,
            "BOOLEAN": Boolean,
            "DATETIME": DateTime,
        }

        columns: list[Column[Any]] = []

        for column in schema:
            if not isinstance(column, dict):
                continue

            name = column.get("name")
            sql_type = column.get("type", "TEXT")

            if not isinstance(name, str) or not name:
                raise ValueError("Invalid column name.")

            sqlalchemy_type = type_map.get(sql_type)

            if sqlalchemy_type is None:
                raise ValueError(
                    f"Unsupported SQL type {sql_type!r} " f"for column {name!r}."
                )

            columns.append(Column(name, sqlalchemy_type, nullable=True))

        if not columns:
            raise ValueError("Schema has no valid columns.")

        return columns
