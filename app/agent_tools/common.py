from __future__ import annotations

import io
import re
import sqlite3
from collections.abc import Sequence

import pandas as pd

from app.database import CHUNK_TABLE_NAME, DOCUMENT_METADATA_TABLE_NAME
from app.store_file.base import FileStorage
from app.store_sql.base import SqlStorage

# Row caps keep tool output small enough for the model's context.
QUERY_TABLE_MAX_ROWS = 5
QUERY_DOCUMENTS_MAX_ROWS = 5

# The main table's name inside sql_query_table's database.
MAIN_TABLE_NAME = "data"

# SQL identifiers the model may use as related-table aliases.
_ALIAS_SHAPE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")

# The main table's source_id, shared by the tools that take it.
SOURCE_ID_DESCRIPTION = (
    "The table's source id, as shown by search_documents or the "
    "inspect_table / sql_query_table results."
)


def object_schema(properties: dict, required: list[str] | None = None) -> dict:
    schema: dict = {"type": "object", "properties": properties}
    if required:
        schema["required"] = required
    return schema


# --- table lookup (no plugin knowledge) ---
#
# The tools know nothing about plugins. A chunk row is a table when its
# metadata carries the table's schema; everything else (the file it came
# from, how to address it) is read from the chunk and document rows.


def is_table_row(row: dict) -> bool:
    metadata = row.get("metadata")
    return isinstance(metadata, dict) and isinstance(metadata.get("schema"), list)


async def table_chunks_for_source(
    sql_storage: SqlStorage, source_id: str, chat_id: str | None = None
) -> list[dict]:
    """All of one source's table chunk rows, chat-scoped when given."""
    def condition(t):
        expr = t.c.source_id == source_id
        if chat_id is not None:
            expr = expr & (t.c.chat_id == chat_id)
        return expr

    rows = await sql_storage.get_all(CHUNK_TABLE_NAME, condition=condition)
    return [row for row in rows if is_table_row(row)]


async def table_chunks_with_origin(
    sql_storage: SqlStorage, origin_source_id: str, chat_id: str | None = None
) -> list[dict]:
    """All table chunk rows whose origin document is the given source."""
    def condition(t):
        expr = t.c.origin_source_id == origin_source_id
        if chat_id is not None:
            expr = expr & (t.c.chat_id == chat_id)
        return expr

    rows = await sql_storage.get_all(CHUNK_TABLE_NAME, condition=condition)
    return [row for row in rows if is_table_row(row)]


async def document_for_source(
    sql_storage: SqlStorage, source_id: str
) -> dict | None:
    return await sql_storage.get(
        DOCUMENT_METADATA_TABLE_NAME,
        condition=lambda t: t.c.source_id == source_id,
    )


def is_csv_document(document: dict) -> bool:
    return (document.get("file_orig_filename") or "").lower().endswith(".csv")


async def resolve_table_source(
    sql_storage: SqlStorage, source_id: str, chat_id: str | None = None
) -> list[dict]:
    """All of a source's table chunk rows, with friendly errors when the
    source is unknown, not in the chat, or stores no table."""
    rows = await table_chunks_for_source(sql_storage, source_id, chat_id)
    if rows:
        return rows
    if chat_id is not None and await table_chunks_for_source(sql_storage, source_id):
        raise ValueError(f"Source '{source_id}''s tables are not in this chat.")
    document = await document_for_source(sql_storage, source_id)
    if document is None:
        raise ValueError(f"Unknown source '{source_id}'.")
    raise ValueError(f"Source '{source_id}' stores no table.")


async def resolve_csv_table(
    sql_storage: SqlStorage, source_id: str, chat_id: str | None = None
) -> tuple[dict, dict]:
    """The (table chunk row, document row) for a source stored as a csv.

    Raises ValueError when the source is unknown, not a csv (tables stored
    as multi-sheet workbooks are not csv tables), or stores no table.
    """
    document = await document_for_source(sql_storage, source_id)
    if document is None:
        raise ValueError(f"Unknown source '{source_id}'.")
    if not is_csv_document(document):
        raise ValueError(
            f"Source '{source_id}' is not a csv table "
            f"(it's a '{document.get('file_orig_filename')}' file)."
        )
    rows = await table_chunks_for_source(sql_storage, source_id, chat_id)
    if not rows:
        if chat_id is not None and await table_chunks_for_source(
            sql_storage, source_id
        ):
            raise ValueError(f"Source '{source_id}''s table is not in this chat.")
        raise ValueError(f"Source '{source_id}' stores no table.")
    if len(rows) > 1:
        raise ValueError(
            f"Source '{source_id}' stores several tables; a csv source "
            "stores one."
        )
    return rows[0], document


# --- table data (sqlite) ---


async def load_csv_table(file_storage: FileStorage, document: dict) -> pd.DataFrame:
    """A csv table's data: the whole file, as a dataframe."""
    return pd.read_csv(io.BytesIO(await file_storage.read_bytes(document["file_path"])))


def validate_table_aliases(aliases: Sequence[str]) -> None:
    for alias in aliases:
        if not _ALIAS_SHAPE.fullmatch(alias):
            raise ValueError(f"Invalid table alias '{alias}'.")
        if alias == MAIN_TABLE_NAME:
            raise ValueError(
                f"'{MAIN_TABLE_NAME}' is reserved for the main table."
            )


def run_table_sql(
    dataframes: dict[str, pd.DataFrame], sql: str
) -> pd.DataFrame:
    """Run one SELECT against the given tables in a throw-away in-memory
    database (read-only by construction; pandas enforces SELECT)."""
    connection = sqlite3.connect(":memory:")
    try:
        for name, dataframe in dataframes.items():
            dataframe.to_sql(name, connection, index=False)
        return pd.read_sql_query(sql, connection)
    finally:
        connection.close()


def format_schema(schema: Sequence[dict]) -> str:
    """The schema as one compact line: `column: TYPE, ...`."""
    if not schema:
        return "(schema unavailable)"
    return ", ".join(f"{col['name']}: {col['type']}" for col in schema)
