from __future__ import annotations

import asyncio
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

from app.core.config import settings
from app.store_metadata.base import MetadataStorage

METADATA_DIR = Path(settings.METADATA_LOCAL_STORAGE_DIR)
METADATA_DATABASE = METADATA_DIR / "metadata.sqlite3"


class LocalMetadataStorage(MetadataStorage):
    """Stores collection-scoped JSON records in a local SQLite database."""

    def __init__(self, database_path: str | Path | None = None):
        self._path = Path(database_path or METADATA_DATABASE)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self._path)

    @contextmanager
    def _connection(self) -> Generator[sqlite3.Connection, None, None]:
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
            connection.execute("""CREATE TABLE IF NOT EXISTS rag_metadata (
                id TEXT PRIMARY KEY,
                collection_name TEXT NOT NULL,
                document_json TEXT NOT NULL
            )""")
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_rag_metadata_collection "
                "ON rag_metadata(collection_name)"
            )

    async def upsert(self, collection_name: str, id: str, document_json: str) -> None:
        """Add or replace one JSON record without blocking the event loop."""
        await asyncio.to_thread(self._upsert, collection_name, id, document_json)

    def _upsert(self, collection_name: str, id: str, document_json: str) -> None:
        with self._connection() as connection:
            connection.execute(
                """INSERT INTO rag_metadata (id, collection_name, document_json)
                VALUES (?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    collection_name = excluded.collection_name,
                    document_json = excluded.document_json""",
                (id, collection_name, document_json),
            )

    async def get(self, id: str) -> str:
        """Retrieve one JSON record by ID."""
        return await asyncio.to_thread(self._get, id)

    def _get(self, id: str) -> str:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT document_json FROM rag_metadata WHERE id = ?", (id,)
            ).fetchone()
        if row is None:
            raise KeyError(id)
        return row[0]

    async def list(self) -> list[str]:
        """List every JSON record."""
        return await asyncio.to_thread(self._list)

    def _list(self) -> list[str]:
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT document_json FROM rag_metadata ORDER BY rowid"
            ).fetchall()
        return [row[0] for row in rows]

    async def delete(self, id: str) -> bool:
        """Delete one JSON record and report whether it existed."""
        return await asyncio.to_thread(self._delete, id)

    def _delete(self, id: str) -> bool:
        with self._connection() as connection:
            cursor = connection.execute("DELETE FROM rag_metadata WHERE id = ?", (id,))
        return cursor.rowcount > 0
