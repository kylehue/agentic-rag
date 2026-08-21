import json
from typing import Any

from sqlalchemy import Boolean, Column, DateTime, Float, Integer, String
from app.models.document import Document, DocumentCategory


def columns_from_document_chunk(chunk: Document) -> list[Column[Any]]:
    """Create SQLAlchemy columns from the authoritative chunk schema."""

    if chunk.category is not DocumentCategory.SPREADSHEET:
        raise ValueError("Chunk is not a spreadsheet chunk.")

    schema = chunk.metadata.get("schema")

    if not isinstance(schema, list) or not schema:
        raise ValueError("Spreadsheet chunk has no schema.")

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

        columns.append(Column(name, sqlalchemy_type))

    if not columns:
        raise ValueError("Spreadsheet chunk has no valid columns.")

    return columns
