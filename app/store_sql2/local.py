from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from sqlalchemy import (
    Column,
    MetaData,
    Table,
    delete,
    inspect,
    select,
    text,
)
from sqlalchemy.dialects.sqlite import insert
from sqlalchemy.exc import NoSuchTableError
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.store_sql2.base import SqlStorage


class LocalSqlStorage(SqlStorage):
    """
    SQLite-backed SQL storage.

    One SQLite database is created per `db_name`.

    Example:

        storage = LocalSqlStorage("storage/sql")

        await storage.ensure_table(
            "documents",
            "documents",
            [
                Column("id", String, primary_key=True),
                Column("filename", String, nullable=False),
            ],
        )

        await storage.upsert(
            "documents",
            "documents",
            [{"id": "123", "filename": "test.pdf"}],
            conflict_columns=["id"],
        )
    """

    def __init__(
        self,
        storage_dir: str | Path,
        *,
        echo: bool = False,
    ) -> None:
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        self.echo = echo

        self._engines: dict[str, AsyncEngine] = {}
        self._sessions: dict[
            str,
            async_sessionmaker[AsyncSession],
        ] = {}

    def _validate_db_name(self, db_name: str) -> None:
        if not db_name:
            raise ValueError("Database name cannot be empty.")

        if Path(db_name).name != db_name:
            raise ValueError(f"Invalid database name: {db_name!r}")

    def _get_engine(self, db_name: str) -> AsyncEngine:
        self._validate_db_name(db_name)

        engine = self._engines.get(db_name)

        if engine is not None:
            return engine

        db_path = self.storage_dir / f"{db_name}.db"

        engine = create_async_engine(
            f"sqlite+aiosqlite:///{db_path}",
            echo=self.echo,
            connect_args={
                "timeout": 30,
            },
            pool_pre_ping=True,
        )

        self._engines[db_name] = engine

        self._sessions[db_name] = async_sessionmaker(
            engine,
            expire_on_commit=False,
        )

        return engine

    def _get_session_factory(
        self,
        db_name: str,
    ) -> async_sessionmaker[AsyncSession]:
        self._get_engine(db_name)

        return self._sessions[db_name]

    async def close(self) -> None:
        """
        Dispose all database connections.

        Call this when shutting down the application.
        """
        for engine in self._engines.values():
            await engine.dispose()

        self._engines.clear()
        self._sessions.clear()

    @staticmethod
    def _validate_identifier(
        value: str,
        *,
        name: str,
    ) -> None:
        """
        Validate SQL identifiers.

        Identifiers cannot safely be passed as bound parameters,
        so table/database names must be validated separately.
        """
        if not value:
            raise ValueError(f"{name} cannot be empty.")

        if len(value) > 255:
            raise ValueError(f"{name} is too long.")

        if not (value[0].isalpha() or value[0] == "_"):
            raise ValueError(f"Invalid {name}: {value!r}")

        if not all(char.isalnum() or char == "_" for char in value):
            raise ValueError(f"Invalid {name}: {value!r}")

    def _validate_table_name(
        self,
        table_name: str,
    ) -> None:
        self._validate_identifier(
            table_name,
            name="table name",
        )

    async def _load_table(
        self,
        db_name: str,
        table_name: str,
    ) -> Table:
        self._validate_table_name(table_name)

        engine = self._get_engine(db_name)

        metadata = MetaData()

        async with engine.connect() as conn:

            def load(sync_conn) -> Table:
                return Table(
                    table_name,
                    metadata,
                    autoload_with=sync_conn,
                )

            try:
                return await conn.run_sync(load)
            except NoSuchTableError:
                raise ValueError(f"Table {table_name!r} does not exist.")

    async def ensure_table(
        self,
        db_name: str,
        table_name: str,
        columns: Sequence[Column[Any]],
    ) -> None:
        """
        Create a table if it doesn't exist.

        Existing tables are left untouched.

        This method does not perform schema migrations.
        """

        self._validate_table_name(table_name)

        if not columns:
            raise ValueError("A table must contain at least one column.")

        metadata = MetaData()

        Table(
            table_name,
            metadata,
            *columns,
        )

        engine = self._get_engine(db_name)

        async with engine.begin() as conn:
            await conn.run_sync(metadata.create_all)

    async def ensure_columns(
        self,
        db_name: str,
        table_name: str,
        columns: Sequence[Column[Any]],
    ) -> None:
        """
        Add missing columns to an existing table.

        SQLite only supports limited ALTER TABLE operations,
        so this intentionally only handles ADD COLUMN.
        """

        self._validate_table_name(table_name)

        engine = self._get_engine(db_name)

        async with engine.begin() as conn:

            def get_existing_columns(sync_conn):
                inspector = inspect(sync_conn)

                return {column["name"] for column in inspector.get_columns(table_name)}

            existing = await conn.run_sync(get_existing_columns)

            for column in columns:
                if column.name in existing:
                    continue

                column_sql = column.compile(dialect=engine.sync_engine.dialect)

                sql = text(
                    f'ALTER TABLE "{table_name}" '
                    f'ADD COLUMN "{column.name}" '
                    f"{column_sql}"
                )

                await conn.execute(sql)

    async def upsert(
        self,
        db_name: str,
        table_name: str,
        rows: Sequence[dict[str, Any]],
        conflict_columns: Sequence[str],
    ) -> None:
        """
        Insert rows or update existing rows.

        The conflict columns must correspond to a PRIMARY KEY
        or UNIQUE constraint/index in SQLite.
        """

        if not rows:
            return

        if not conflict_columns:
            raise ValueError("At least one conflict column is required.")

        table = await self._load_table(
            db_name,
            table_name,
        )

        table_columns = {column.name for column in table.columns}

        for column in conflict_columns:
            if column not in table_columns:
                raise ValueError(f"Unknown conflict column: {column!r}")

        # Prevent accidental insertion of columns that don't
        # exist in the database schema.
        for row in rows:
            unknown = set(row) - table_columns

            if unknown:
                raise ValueError(f"Unknown columns: {sorted(unknown)}")

        statement = insert(table).values(list(rows))

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

        session_factory = self._get_session_factory(db_name)

        async with session_factory() as session:
            async with session.begin():
                await session.execute(statement)

    async def get(
        self,
        db_name: str,
        table_name: str,
        row_id: str,
    ) -> dict[str, Any] | None:
        table = await self._load_table(
            db_name,
            table_name,
        )

        if "id" not in table.c:
            raise ValueError(f"Table {table_name!r} has no 'id' column.")

        statement = select(table).where(table.c.id == row_id).limit(1)

        session_factory = self._get_session_factory(db_name)

        async with session_factory() as session:
            result = await session.execute(statement)

            row = result.mappings().first()

            if row is None:
                return None

            return dict(row)

    async def get_all(
        self,
        db_name: str,
        table_name: str,
    ) -> Sequence[dict[str, Any]]:
        table = await self._load_table(
            db_name,
            table_name,
        )

        statement = select(table)

        session_factory = self._get_session_factory(db_name)

        async with session_factory() as session:
            result = await session.execute(statement)

            return [dict(row) for row in result.mappings()]

    async def delete(
        self,
        db_name: str,
        table_name: str,
        row_id: str,
    ) -> bool:
        table = await self._load_table(
            db_name,
            table_name,
        )

        if "id" not in table.c:
            raise ValueError(f"Table {table_name!r} has no 'id' column.")

        statement = delete(table).where(table.c.id == row_id)

        session_factory = self._get_session_factory(db_name)

        async with session_factory() as session:
            async with session.begin():
                result = await session.execute(statement)

            return result.scalar_one_or_none() is not None

    async def query(
        self,
        db_name: str,
        sql_query: str,
        limit: int,
    ) -> Sequence[dict[str, Any]]:
        """
        Execute a read-only SQL query.

        IMPORTANT:
        This is intended for trusted/generated SELECT statements.
        For LLM-generated SQL, additional SQL parsing/validation
        should be performed before reaching this method.
        """
        if limit <= 0:
            return []

        query = sql_query.strip()

        if not query:
            raise ValueError("SQL query cannot be empty.")

        # Basic first-line defense.
        if not query.lower().startswith("select"):
            raise ValueError("Only SELECT statements are allowed.")

        # Prevent multiple statements.
        if ";" in query.rstrip(";"):
            raise ValueError("Multiple SQL statements are not allowed.")

        statement = text(query)

        session_factory = self._get_session_factory(db_name)

        async with session_factory() as session:
            result = await session.execute(statement)

            rows = result.mappings().fetchmany(limit)

            return [dict(row) for row in rows]

    async def search(
        self,
        db_name: str,
        table_name: str,
        search_query: str,
        limit: int,
    ) -> Sequence[dict[str, Any]]:
        """
        Search an FTS5 virtual table.

        Expected convention:

            <table_name>_fts

        Example:

            documents
            documents_fts
        """

        if limit <= 0:
            return []

        self._validate_table_name(table_name)

        fts_table = f"{table_name}_fts"

        statement = text(f"""
            SELECT *
            FROM "{fts_table}"
            WHERE "{fts_table}" MATCH :query
            LIMIT :limit
            """)

        session_factory = self._get_session_factory(db_name)

        async with session_factory() as session:
            result = await session.execute(
                statement,
                {
                    "query": search_query,
                    "limit": limit,
                },
            )

            return [dict(row) for row in result.mappings()]
