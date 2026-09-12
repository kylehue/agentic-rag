from __future__ import annotations

import io
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from app.core.config import settings
from app.store_file.base import FileStorage
from app.store_sql.base import SqlStorage

# Row caps keep tool output small enough for the model's context.
QUERY_TABLE_MAX_ROWS = 100

# How many rows to read when inferring a table's schema from its file (the
# fallback for tables ingested before the schema was stored in metadata).
SCHEMA_INFER_ROWS = 20


def object_schema(properties: dict, required: list[str] | None = None) -> dict:
    schema: dict = {"type": "object", "properties": properties}
    if required:
        schema["required"] = required
    return schema


# --- table resolution ---


async def table_rows(sql_storage: SqlStorage) -> Sequence[dict]:
    return await sql_storage.get_all(
        settings.CHUNK_TABLE_NAME,
        condition=lambda t: t.c.plugin == "table",
    )


async def resolve_table_row(
    sql_storage: SqlStorage,
    table_name: str,
    source_id: str | None = None,
) -> dict:
    """The stored chunk row for a table by its original name.

    When several sources store a table with the same name, `source_id`
    (as shown by list_tables) disambiguates; otherwise the collision is an
    error so the wrong table is never used silently.
    """
    matches = [
        row
        for row in await table_rows(sql_storage)
        if (row.get("metadata") or {}).get("table_name") == table_name
    ]
    if source_id is not None:
        matches = [row for row in matches if row["source_id"] == source_id]
    if not matches:
        if source_id is None:
            raise ValueError(f"No table named '{table_name}' is stored.")
        raise ValueError(
            f"No table named '{table_name}' is stored under source '{source_id}'."
        )
    if len(matches) > 1:
        sources = ", ".join(sorted({row["source_id"] for row in matches}))
        raise ValueError(
            f"Table name '{table_name}' is ambiguous (stored under sources: "
            f"{sources}). Pass the source_id shown by list_tables to "
            "disambiguate."
        )
    return matches[0]


async def resolve_table_file(
    sql_storage: SqlStorage,
    file_storage: FileStorage,
    table_name: str,
    source_id: str | None = None,
) -> tuple[str, bytes]:
    """The stored file (path + bytes) for a table. Used only when the table's
    data must actually be read (running a query), never for inspection."""
    row = await resolve_table_row(sql_storage, table_name, source_id)
    row_source_id = row["source_id"]
    document = await sql_storage.get(
        settings.DOCUMENT_METADATA_TABLE_NAME,
        condition=lambda t: t.c.source_id == row_source_id,
    )
    if document is None:
        raise ValueError(f"Table '{table_name}' has no stored source file.")
    file_path = document["file_path"]
    return file_path, await file_storage.read_bytes(file_path)


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
    table_name: str, file_path: str, file_bytes: bytes, nrows: int
) -> pd.DataFrame:
    """Read at most `nrows` rows of the table's sheet — a bounded peek, not
    the whole file."""
    extension = Path(file_path).suffix.lower()
    source = io.BytesIO(file_bytes)
    if extension == ".csv":
        return pd.read_csv(source, nrows=nrows)
    if extension in {".xlsx", ".xls"}:
        return pd.read_excel(source, sheet_name=table_name, nrows=nrows)
    raise ValueError(f"Unsupported table file type: '{extension}'")


def read_sheet(table_name: str, file_path: str, file_bytes: bytes) -> pd.DataFrame:
    """Read the table's full sheet (only sheet, not the whole workbook)."""
    extension = Path(file_path).suffix.lower()
    source = io.BytesIO(file_bytes)
    if extension == ".csv":
        return pd.read_csv(source)
    if extension in {".xlsx", ".xls"}:
        return pd.read_excel(source, sheet_name=table_name)
    raise ValueError(f"Unsupported table file type: '{extension}'")


def format_schema(schema: Sequence[dict]) -> str:
    if not schema:
        return "  (schema unavailable)"
    return "\n".join(f"  {col['name']}: {col['type']}" for col in schema)
