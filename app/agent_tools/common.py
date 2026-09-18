from __future__ import annotations

import io
import sqlite3
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from app.database import CHUNK_TABLE_NAME, DOCUMENT_METADATA_TABLE_NAME
from app.store_file.base import FileStorage
from app.store_sql.base import SqlStorage

# Row caps keep tool output small enough for the model's context.
SQL_MAX_ROWS = 5

# The stored-file extensions that mark a source as tabular data.
TABLE_EXTENSIONS = frozenset({"csv", "xlsx", "xls"})

# The shared description for a source_id parameter, so the tools agree. It is
# deliberately emphatic about what the id is NOT: the model otherwise tends to
# substitute a file name or a table name for it.
SOURCE_ID_PARAMETER = (
    "The document's source id: an opaque UUID-like identifier, shown as "
    "`source_id=` in search results. Copy that exact value; it is not the "
    "file name and not a table name."
)


def object_schema(properties: dict, required: list[str] | None = None) -> dict:
    schema: dict = {"type": "object", "properties": properties}
    if required:
        schema["required"] = required
    return schema


# --- source / table lookup (no plugin knowledge) ---
#
# A chunk row is a table when its metadata carries the table's schema; the
# stored file a source points at is read from the document row.


def is_table_row(row: dict) -> bool:
    metadata = row.get("metadata")
    return isinstance(metadata, dict) and isinstance(metadata.get("schema"), list)


def document_extension(document: dict) -> str:
    """The stored file's extension (lowercased, no dot), from its original
    filename, falling back to the stored name."""
    for key in ("file_orig_filename", "file_filename"):
        name = document.get(key)
        if name:
            return Path(name).suffix.lower().lstrip(".")
    return ""


def is_table_type(document: dict) -> bool:
    return document_extension(document) in TABLE_EXTENSIONS


async def document_for_source(
    sql_storage: SqlStorage, source_id: str
) -> dict | None:
    return await sql_storage.get(
        DOCUMENT_METADATA_TABLE_NAME,
        condition=lambda t: t.c.source_id == source_id,
    )


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


async def resolve_table_document(
    sql_storage: SqlStorage, source_id: str, chat_id: str | None = None
) -> dict:
    """The stored document for a table source, with friendly errors when the
    source is unknown, not in this chat, or not a spreadsheet."""
    document = await document_for_source(sql_storage, source_id)
    if document is None:
        raise ValueError(f"Unknown source '{source_id}'.")
    if chat_id is not None and document.get("chat_id") != chat_id:
        raise ValueError(f"Source '{source_id}' is not in this chat.")
    if not is_table_type(document):
        raise ValueError(
            f"Source '{source_id}' is not a table "
            f"(it's a '{document_extension(document) or 'unknown'}' file)."
        )
    return document


def table_schema_entries(rows: Sequence[dict]) -> list[dict]:
    """One entry per table chunk: its table name, schema, and row count, read
    from the stored metadata (no data read)."""
    entries = []
    for row in rows:
        metadata = row.get("metadata") or {}
        entries.append(
            {
                "table_name": metadata.get("table_name"),
                "schema": metadata.get("schema"),
                "row_count": metadata.get("row_count"),
            }
        )
    return entries


# --- table data (sqlite) ---


async def read_source_tables(
    file_storage: FileStorage, document: dict
) -> dict[str, pd.DataFrame]:
    """A spreadsheet source's data as {table name: DataFrame}. A csv is one
    table (named after the file); a workbook is one table per sheet (named
    after the sheet). The names match the table_name stored in the chunks'
    metadata, so they are what the model uses in SQL and for JOINs."""
    raw = await file_storage.read_bytes(document["file_path"])
    extension = document_extension(document)
    if extension == "csv":
        name = Path(document.get("file_orig_filename") or document.get("file_filename") or "").stem
        return {name or "data": pd.read_csv(io.BytesIO(raw))}
    if extension in ("xlsx", "xls"):
        sheets = pd.read_excel(io.BytesIO(raw), sheet_name=None)
        return {str(name): dataframe for name, dataframe in sheets.items()}
    raise ValueError(f"Source is not a spreadsheet (it's a '{extension}' file).")


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
