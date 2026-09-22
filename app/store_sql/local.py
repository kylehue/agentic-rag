from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
import re
from typing import Any

from sqlalchemy import MetaData, Table, delete, select, text
from sqlalchemy.dialects.sqlite import insert
from sqlalchemy.exc import NoSuchTableError
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    async_sessionmaker,
    create_async_engine,
)

from app.database import Base
from app.store_sql.base import ConditionBuilder, SqlStorage


def _fts_query(value: str) -> str:
    tokens = re.findall(r"\w+", value, flags=re.UNICODE)

    if not tokens:
        return ""

    quoted_tokens = []

    for token in tokens:
        escaped = token.replace('"', '""')
        quoted_tokens.append(f'"{escaped}"')

    return " OR ".join(quoted_tokens)


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

    async def _ensure_fts_table(self, table_name: str) -> None:
        """
        Lazily create and synchronize an FTS5 table for a normal SQL table.

        The source table must contain:
            - id
            - text

        FTS5 uses the source table as external content, so the actual text
        remains stored only in the source table.
        """

        self._validate_table_name(table_name)

        table = await self._load_table(table_name)

        if "id" not in table.c:
            raise ValueError(
                f"Table {table_name!r} must have an 'id' column " "to use FTS5."
            )

        if "text" not in table.c:
            raise ValueError(
                f"Table {table_name!r} must have a 'text' column " "to use FTS5."
            )

        fts_table = f"{table_name}_fts"
        insert_trigger = f"{fts_table}_ai"
        delete_trigger = f"{fts_table}_ad"
        update_trigger = f"{fts_table}_au"

        async with self._engine.begin() as conn:
            await conn.execute(text(f"""
                    CREATE VIRTUAL TABLE IF NOT EXISTS "{fts_table}"
                    USING fts5(
                        text,
                        content="{table_name}",
                        content_rowid="id"
                    )
                    """))

            await conn.execute(text(f"""
                    CREATE TRIGGER IF NOT EXISTS "{insert_trigger}"
                    AFTER INSERT ON "{table_name}"
                    BEGIN
                        INSERT INTO "{fts_table}"(rowid, text)
                        VALUES (new.id, new.text);
                    END
                    """))

            await conn.execute(text(f"""
                    CREATE TRIGGER IF NOT EXISTS "{delete_trigger}"
                    AFTER DELETE ON "{table_name}"
                    BEGIN
                        INSERT INTO "{fts_table}"(
                            "{fts_table}",
                            rowid,
                            text
                        )
                        VALUES (
                            'delete',
                            old.id,
                            old.text
                        );
                    END
                    """))

            await conn.execute(text(f"""
                    CREATE TRIGGER IF NOT EXISTS "{update_trigger}"
                    AFTER UPDATE OF text ON "{table_name}"
                    BEGIN
                        INSERT INTO "{fts_table}"(
                            "{fts_table}",
                            rowid,
                            text
                        )
                        VALUES (
                            'delete',
                            old.id,
                            old.text
                        );

                        INSERT INTO "{fts_table}"(
                            rowid,
                            text
                        )
                        VALUES (
                            new.id,
                            new.text
                        );
                    END
                    """))

            # Synchronize rows that existed before the FTS table was created.
            await conn.execute(text(f"""
                    INSERT INTO "{fts_table}"("{fts_table}")
                    VALUES ('rebuild')
                    """))

    async def close(self) -> None:
        await self._engine.dispose()

    def get_sql_dialect(self) -> str:
        return "SQLite"

    async def get_table(self, table_name: str) -> Table:
        return await self._load_table(table_name)

    async def create_tables(self) -> None:
        """Create every table defined on the database base, if it is absent.

        The schema is owned by the ORM models in app.database; this applies
        it. New tables are created whole; for tables that already exist, any
        missing *nullable* column is added in place (a schema addition, like a
        new optional chunk column), so an existing database picks it up without
        a wipe. Non-nullable or primary-key columns are never added in place.
        """
        async with self._engine.begin() as conn:
            await conn.run_sync(self._create_and_migrate)

    def _create_and_migrate(self, conn) -> None:
        from sqlalchemy import inspect as sa_inspect

        Base.metadata.create_all(conn)
        inspector = sa_inspect(conn)
        for table in Base.metadata.sorted_tables:
            try:
                existing = {c["name"] for c in inspector.get_columns(table.name)}
            except NoSuchTableError:
                continue  # just created by create_all, already whole
            for column in table.columns:
                if column.name in existing or column.primary_key:
                    continue
                if not column.nullable:
                    continue
                coltype = column.type.compile(dialect=conn.dialect)
                conn.execute(
                    text(
                        f'ALTER TABLE "{table.name}" '
                        f'ADD COLUMN "{column.name}" {coltype}'
                    )
                )

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
        condition: ConditionBuilder,
    ) -> dict[str, Any] | None:
        if condition is None:
            raise ValueError("Get requires a condition.")

        table = await self.get_table(table_name)

        statement = select(table).where(condition(table)).limit(1)

        async with self._session_factory() as session:
            result = await session.execute(statement)

            row = result.mappings().first()

            if row is None:
                return None

            return dict(row)

    async def get_all(
        self,
        table_name: str,
        condition: ConditionBuilder | None = None,
        limit: int | None = None,
    ) -> Sequence[dict[str, Any]]:
        table = await self.get_table(table_name)

        if limit is not None and limit <= 0:
            return []

        statement = select(table)

        if condition is not None:
            statement = statement.where(condition(table))

        if limit is not None:
            statement = statement.limit(limit)

        async with self._session_factory() as session:
            result = await session.execute(statement)

            return [dict(row) for row in result.mappings()]

    async def delete(
        self,
        table_name: str,
        condition: ConditionBuilder,
    ) -> bool:
        table = await self.get_table(table_name)

        statement = delete(table).where(condition(table)).returning(next(iter(table.c)))

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
        condition: ConditionBuilder | None = None,
    ) -> Sequence[dict[str, Any]]:
        self._validate_table_name(table_name)

        if not search_query.strip():
            return []

        if limit is not None and limit <= 0:
            return []

        fts_query = _fts_query(search_query)

        if not fts_query:
            return []

        await self._ensure_fts_table(table_name)

        fts_table = f"{table_name}_fts"
        limit_clause = "" if limit is None else "LIMIT :limit"

        condition_clause = ""
        if condition is not None:
            # Rendered with literal binds: SQLAlchemy quotes the values, and
            # the conditions in use are system-generated column filters.
            table = await self._load_table(table_name)
            rendered = condition(table).compile(
                dialect=self._engine.dialect,
                compile_kwargs={"literal_binds": True},
            )
            condition_clause = f"AND {rendered}"

        # The table is referenced by its real name (not an alias) so that
        # the compiled condition, which is built against the real table,
        # resolves against this query.
        statement = text(f"""
            SELECT "{table_name}".*
            FROM "{table_name}"
            JOIN "{fts_table}" AS fts
                ON "{table_name}".id = fts.rowid
            WHERE "{fts_table}" MATCH :query
            {condition_clause}
            ORDER BY bm25("{fts_table}")
            {limit_clause}
            """)

        parameters: dict[str, Any] = {
            "query": fts_query,
        }

        if limit is not None:
            parameters["limit"] = limit

        async with self._session_factory() as session:
            result = await session.execute(
                statement,
                parameters,
            )

            return [dict(row) for row in result.mappings()]
