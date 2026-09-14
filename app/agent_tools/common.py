from __future__ import annotations

import io
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from app.core.config import settings
from app.store_file.base import FileStorage
from app.store_sql.base import SqlStorage

# Row caps keep tool output small enough for the model's context.
QUERY_TABLE_MAX_ROWS = 5

# How many rows to read when inferring a table's schema from its file (the
# fallback for tables ingested before the schema was stored in metadata).
SCHEMA_INFER_ROWS = 20

# What the table-addressing parameters mean, shared by the tools that take
# them (inspect_table, query_table).
SOURCE_ID_DESCRIPTION = "The table's source id, as shown by list_tables."
CHUNK_ID_DESCRIPTION = (
    "Only when the source stores several tables (e.g. a workbook with "
    "multiple sheets): the table's chunk id, as shown by list_tables."
)


def object_schema(properties: dict, required: list[str] | None = None) -> dict:
    schema: dict = {"type": "object", "properties": properties}
    if required:
        schema["required"] = required
    return schema


# --- table resolution ---


async def table_rows(
    sql_storage: SqlStorage, chat_id: str | None = None
) -> Sequence[dict]:
    def condition(t):
        expr = t.c.plugin == "table"
        if chat_id is not None:
            expr = expr & (t.c.chat_id == chat_id)
        return expr

    return await sql_storage.get_all(settings.CHUNK_TABLE_NAME, condition=condition)


async def resolve_table(
    sql_storage: SqlStorage,
    source_id: str,
    chunk_id: str | None = None,
    chat_id: str | None = None,
) -> dict:
    """The stored chunk row for one table, addressed by its source (the file
    it came from) and, when that source stores several tables (e.g. the
    sheets of a workbook), the specific table's chunk id.

    The table's own name never identifies it: it is read from the resolved
    row's metadata. `chat_id`, when given, bounds the search to that chat's
    tables.
    """
    rows = [
        row
        for row in await table_rows(sql_storage, chat_id)
        if row["source_id"] == source_id
    ]
    if chunk_id is not None:
        rows = [row for row in rows if row["chunk_id"] == chunk_id]

    if len(rows) > 1:
        names = ", ".join(
            f"'{(row.get('metadata') or {}).get('table_name', row['chunk_id'])}'"
            for row in rows
        )
        raise ValueError(
            f"Source '{source_id}' stores several tables ({names}). Pass the "
            "chunk_id shown by list_tables to pick one."
        )

    if not rows:
        if chunk_id is not None:
            raise ValueError(
                f"Source '{source_id}' has no table with chunk_id '{chunk_id}'."
            )
        document = await sql_storage.get(
            settings.DOCUMENT_METADATA_TABLE_NAME,
            condition=lambda t: t.c.source_id == source_id,
        )
        if document is None:
            raise ValueError(f"Unknown source '{source_id}'.")
        # The source exists but has no table row in scope: either it isn't a
        # table at all, or its tables live in another chat.
        if chat_id is not None and any(
            row["source_id"] == source_id for row in await table_rows(sql_storage)
        ):
            raise ValueError(f"Source '{source_id}''s tables are not in this chat.")
        raise ValueError(
            f"Source '{source_id}' isn't a table "
            f"(it's a {document['file_content_type']} document)."
        )

    return rows[0]


async def resolve_table_file(
    sql_storage: SqlStorage,
    file_storage: FileStorage,
    source_id: str,
    chunk_id: str | None = None,
    chat_id: str | None = None,
) -> tuple[str, bytes, str]:
    """The stored file (path + bytes) and sheet name for one table. Used only
    when the table's data must actually be read (running a query, inferring
    a schema), never for inspection from metadata."""
    row = await resolve_table(sql_storage, source_id, chunk_id, chat_id)
    document = await sql_storage.get(
        settings.DOCUMENT_METADATA_TABLE_NAME,
        condition=lambda t: t.c.source_id == row["source_id"],
    )
    if document is None:
        raise ValueError(
            f"Table under source '{source_id}' has no stored source file."
        )
    sheet = (row.get("metadata") or {}).get("table_name") or ""
    return document["file_path"], await file_storage.read_bytes(
        document["file_path"]
    ), sheet


def pandas_dtype_to_type(dtype) -> str:
    if pd.api.types.is_bool_dtype(dtype):
        return "BOOLEAN"
    if pd.api.types.is_integer_dtype(dtype):
        return "INTEGER"
    if pd.api.types.is_float_dtype(dtype):
        return "REAL"
    if pd.api.types.is_datetime64_any_dtype(dtype):
        return "DATETIME"
    return "TEXT"


def dataframe_schema(dataframe: pd.DataFrame) -> list[dict]:
    return [
        {"name": str(column), "type": pandas_dtype_to_type(dataframe[column].dtype)}
        for column in dataframe.columns
    ]


def read_sheet_sample(
    sheet_name: str, file_path: str, file_bytes: bytes, nrows: int
) -> pd.DataFrame:
    """Read at most `nrows` rows of the table's sheet — a bounded peek, not
    the whole file."""
    extension = Path(file_path).suffix.lower()
    source = io.BytesIO(file_bytes)
    if extension == ".csv":
        return pd.read_csv(source, nrows=nrows)
    if extension in {".xlsx", ".xls"}:
        return pd.read_excel(source, sheet_name=sheet_name, nrows=nrows)
    raise ValueError(f"Unsupported table file type: '{extension}'")


def read_sheet(sheet_name: str, file_path: str, file_bytes: bytes) -> pd.DataFrame:
    """Read the table's full sheet (only sheet, not the whole workbook)."""
    extension = Path(file_path).suffix.lower()
    source = io.BytesIO(file_bytes)
    if extension == ".csv":
        return pd.read_csv(source)
    if extension in {".xlsx", ".xls"}:
        return pd.read_excel(source, sheet_name=sheet_name)
    raise ValueError(f"Unsupported table file type: '{extension}'")


def format_schema(schema: Sequence[dict]) -> str:
    if not schema:
        return "  (schema unavailable)"
    return "\n".join(f"  {col['name']}: {col['type']}" for col in schema)
