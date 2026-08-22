from __future__ import annotations

import asyncio
from dataclasses import dataclass
from io import BytesIO
import json
from pathlib import Path
import re
from typing import Any

import pandas as pd
from unstructured.documents.elements import Element, Table

from app.models.document import Document, DocumentCategory
from app.models.ingestion import ProcessorPayload

MAX_WORKBOOK_CONTEXT_CHARS = 30_000
MAX_TABLE_CONTEXT_CHARS = 10_000
MAX_TEXT_CONTEXT_CHARS = 8_000
SAMPLED_DATA_ROWS_PER_TABLE = 5


WORKBOOK_ANALYSIS_PROMPT = """You are analyzing a workbook/document for a retrieval system.

The input is a catalog of tables extracted from one source. Tables may refer to,
define, summarize, or join other tables. Analyze the catalog as a whole so each
table's meaning is informed by the other tables.

Return valid JSON only in this exact shape:
{
  "workbook_description": "one concise description of the source",
  "tables": [
    {
      "index": 0,
      "description": "what this table contains and its role in the source",
      "role": "for example: lookup, fact data, summary, instructions, assumptions",
      "schema": [
        {
          "name": "column_name",
          "description": "meaning of the column"
        }
      ],
      "relationships": [
        {
          "table_index": 1,
          "relationship": "how this table relates to that table"
        }
      ]
    }
  ]
}

Include every catalog index exactly once.

The column data types are already inferred from the source data and are provided
separately. Do not infer or modify data types.

Use an empty relationships list when no relationship is supported by the data.
Do not invent joins, formulas, or facts.

Nearby extracted document text is additional context only. Distinguish it from
the actual table data and do not assume that every statement in the context
belongs to the table.

Nearby text context:
{context}

Table catalog:
{catalog}
"""


@dataclass(frozen=True)
class ExtractedTable:
    """A table represented independently of Unstructured's Table element."""

    dataframe: pd.DataFrame
    name: str
    file_filename: str
    file_content_type: str
    file_bytes: bytes | None = None
    page_number: int | None = None
    orig_elements: list[Element] = []


def _text_context(elements: list[Element]) -> str:
    """Collect extracted text to help the LLM understand table meaning."""

    texts: list[str] = []

    for element in elements:
        text = getattr(element, "text", None)

        if isinstance(text, str) and text.strip():
            texts.append(text.strip())

    context = "\n\n".join(texts)

    return context[:MAX_TEXT_CONTEXT_CHARS]


def _table_label(
    table: Table,
    index: int,
) -> str:
    """Return a stable human-readable label for a PDF table."""

    return getattr(table.metadata, "page_name", None) or f"Table {index + 1}"


def _table_html(table: Table) -> str:
    """Return Unstructured's reconstructed table HTML."""

    value = getattr(
        table.metadata,
        "text_as_html",
        None,
    )

    return value if isinstance(value, str) else ""


def _normalize_name(name: Any) -> str:
    """Convert a name into a safe SQL-friendly snake_case identifier."""

    name = str(name).strip()

    # Replace non-alphanumeric characters with underscores.
    name = re.sub(r"[^a-zA-Z0-9]+", "_", name)

    # Remove leading/trailing underscores.
    name = name.strip("_").lower()

    # Avoid empty identifiers.
    if not name:
        name = "value"

    # Avoid identifiers starting with a number.
    if name[0].isdigit():
        name = f"value_{name}"

    return name


def _normalize_unique_names(
    names: list[Any],
) -> list[str]:
    """Normalize names and ensure every resulting name is unique."""

    used: set[str] = set()
    normalized: list[str] = []

    for value in names:
        base_name = _normalize_name(value)

        name = base_name
        suffix = 2

        while name in used:
            name = f"{base_name}_{suffix}"
            suffix += 1

        used.add(name)
        normalized.append(name)

    return normalized


def _normalize_dataframe_columns(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Normalize DataFrame columns into unique SQL-safe names."""

    dataframe = dataframe.copy()

    dataframe.columns = _normalize_unique_names(list(dataframe.columns))

    return dataframe


def _normalize_table_names(
    names: list[Any],
    file_id: str,
) -> list[str]:
    """Prefix table names with a file ID, then normalize and deduplicate them."""

    return _normalize_unique_names([f"{file_id}_{name}" for name in names])


def _dataframe_from_html(
    table: Table,
) -> pd.DataFrame:
    """Convert an Unstructured PDF table into a DataFrame."""

    html = _table_html(table)

    if not html:
        raise ValueError("Table does not contain text_as_html.")

    tables = pd.read_html(html)

    if not tables:
        raise ValueError("No table could be parsed from text_as_html.")

    return tables[0]


def _read_document_tables(
    payload: ProcessorPayload,
) -> list[ExtractedTable]:
    """
    Extract PDF tables using Unstructured HTML, then immediately convert
    them into DataFrames.
    """

    unstructured_tables = [
        element for element in payload.elements if isinstance(element, Table)
    ]

    if not unstructured_tables:
        return []

    source_names = [
        _table_label(table, index) for index, table in enumerate(unstructured_tables)
    ]

    table_names = _normalize_table_names(
        source_names,
        payload.source_id,
    )

    tables: list[ExtractedTable] = []

    for table, table_name, source_name in zip(
        unstructured_tables,
        table_names,
        source_names,
    ):
        dataframe = _dataframe_from_html(table)
        dataframe = _normalize_dataframe_columns(dataframe)

        tables.append(
            ExtractedTable(
                dataframe=dataframe,
                file_filename=f"{source_name}.csv",
                file_content_type="text/csv",
                name=table_name,
                page_number=getattr(
                    table.metadata,
                    "page_number",
                    None,
                ),
                orig_elements=[table],
            )
        )

    return tables


def _read_spreadsheet_tables(
    payload: ProcessorPayload,
) -> list[ExtractedTable]:
    """
    Read spreadsheet data directly with pandas.

    CSV produces one table.
    XLS/XLSX produces one table per sheet.
    """

    extension = Path(payload.source_filename).suffix.lower()

    source = BytesIO(payload.source_bytes)

    if extension == ".csv":
        dataframe = pd.read_csv(source)
        dataframe = _normalize_dataframe_columns(dataframe)

        source_name = Path(payload.source_filename).stem

        table_name = _normalize_table_names(
            [source_name],
            payload.source_id,
        )[0]

        return [
            ExtractedTable(
                dataframe=dataframe,
                file_filename=payload.source_filename,
                file_bytes=payload.source_bytes,
                file_content_type=payload.source_content_type,
                name=table_name,
                orig_elements=payload.elements,
            )
        ]

    if extension in {".xlsx", ".xls"}:
        sheets = pd.read_excel(
            source,
            sheet_name=None,
        )

        source_names = list(sheets.keys())

        table_names = _normalize_table_names(
            source_names,
            payload.source_id,
        )

        result: list[ExtractedTable] = []

        for (
            source_name,
            table_name,
        ) in zip(
            source_names,
            table_names,
        ):
            dataframe = _normalize_dataframe_columns(sheets[source_name])

            result.append(
                ExtractedTable(
                    dataframe=dataframe,
                    file_filename=payload.source_filename,
                    file_bytes=payload.source_bytes,
                    file_content_type=payload.source_content_type,
                    name=table_name,
                    orig_elements=payload.elements,
                )
            )

        return result

    return []


def _extract_tables(payload: ProcessorPayload) -> list[ExtractedTable]:
    """
    Extract tables according to the source category.

    Documents:
        Unstructured Table -> metadata.text_as_html -> DataFrame

    Spreadsheets:
        Raw file bytes -> pandas -> DataFrame(s)
    """

    if payload.category is DocumentCategory.DOCUMENT:
        return _read_document_tables(payload)

    if payload.category is DocumentCategory.SPREADSHEET:
        return _read_spreadsheet_tables(payload)

    return []


def _pandas_dtype_to_sql_type(dtype: Any) -> str:
    """
    Convert a pandas dtype into the SQL type used by the application.

    The types are intentionally conservative because the DataFrame is the
    source of truth for what will actually be inserted into SQL.
    """

    if pd.api.types.is_bool_dtype(dtype):
        return "BOOLEAN"
    if pd.api.types.is_integer_dtype(dtype):
        return "INTEGER"
    if pd.api.types.is_float_dtype(dtype):
        return "REAL"
    if pd.api.types.is_datetime64_any_dtype(dtype):
        return "DATETIME"

    return "TEXT"


def _schema_for_prompt(dataframe: pd.DataFrame) -> list[dict[str, str]]:
    """
    Build the schema representation sent to the LLM.

    The LLM receives the already-inferred SQL type and only needs to explain
    the semantic meaning of the column.
    """

    return [
        {
            "name": str(column),
            "type": _pandas_dtype_to_sql_type(dataframe[column].dtype),
            "description": "",
        }
        for column in dataframe.columns
    ]


def _sample_table_rows(dataframe: pd.DataFrame, char_limit: int) -> str:
    """
    Produce a compact preview containing column names and a few representative
    rows from the DataFrame.
    """

    if dataframe.empty:
        return "[empty table]"

    sample = dataframe.head(SAMPLED_DATA_ROWS_PER_TABLE)

    headers = " | ".join(str(column) for column in dataframe.columns)

    lines = [headers]

    for row in sample.itertuples(index=False, name=None):
        lines.append(" | ".join(str(value) for value in row))

    preview = "\n".join(lines)

    if len(preview) > char_limit:
        return preview[:char_limit] + "\n[preview truncated]"

    return preview


def _build_catalog(tables: list[ExtractedTable]) -> str:
    """
    Create a row-sampled catalog containing each table's schema, data preview,
    and location information.
    """

    if not tables:
        return "[]"

    per_table_limit = min(
        MAX_TABLE_CONTEXT_CHARS,
        max(1, MAX_WORKBOOK_CONTEXT_CHARS // len(tables)),
    )

    catalog: list[dict[str, Any]] = []

    for index, table in enumerate(tables):
        catalog.append(
            {
                "index": index,
                "table_name": table.name,
                "page_number": table.page_number,
                "row_count": len(table.dataframe),
                "column_count": len(table.dataframe.columns),
                "schema": _schema_for_prompt(table.dataframe),
                "data_preview": _sample_table_rows(
                    table.dataframe,
                    per_table_limit,
                ),
            }
        )

    return json.dumps(
        catalog,
        ensure_ascii=False,
    )


def _parse_json(response: str) -> Any:
    """Accept JSON wrapped in a Markdown code fence."""

    response = response.strip()

    if response.startswith("```"):
        response = response.split("\n", 1)[-1]

        if response.rstrip().endswith("```"):
            response = response.rstrip()[:-3].rstrip()

    return json.loads(response)


def _empty_analysis() -> dict[str, Any]:
    return {
        "description": "",
        "role": "",
        "schema": [],
        "relationships": [],
    }


def _parse_workbook_analysis(
    response: str,
    table_count: int,
) -> tuple[str, list[dict[str, Any]]]:
    """Normalize model output and preserve missing table analyses."""

    payload = _parse_json(response)

    if not isinstance(payload, dict):
        raise ValueError("Workbook analysis must be a JSON object.")

    workbook_description = payload.get("workbook_description", "")

    if not isinstance(workbook_description, str):
        workbook_description = str(workbook_description)

    analyses = [_empty_analysis() for _ in range(table_count)]

    table_analyses = payload.get(
        "tables",
        [],
    )

    if not isinstance(table_analyses, list):
        return workbook_description, analyses

    for item in table_analyses:
        if not isinstance(item, dict):
            continue

        index = item.get("index")

        if not isinstance(index, int):
            continue

        if not 0 <= index < table_count:
            continue

        description = item.get("description", "")
        role = item.get("role", "")
        schema = item.get("schema", [])
        relationships = item.get("relationships", [])
        analyses[index] = {
            "description": (
                description if isinstance(description, str) else str(description)
            ),
            "role": (role if isinstance(role, str) else str(role)),
            "schema": (schema if isinstance(schema, list) else []),
            "relationships": (relationships if isinstance(relationships, list) else []),
        }

    return workbook_description, analyses


def _merge_schema_types(
    dataframe: pd.DataFrame,
    analysis_schema: list[Any],
) -> list[dict[str, str]]:
    """
    Merge LLM-generated column descriptions with authoritative pandas types.

    Pandas determines the type.
    LLM determines the semantic description.
    """

    descriptions: dict[str, str] = {}

    for column in analysis_schema:
        if not isinstance(column, dict):
            continue

        name = column.get("name")

        if not isinstance(name, str):
            continue

        description = column.get("description", "")
        descriptions[name] = (
            description if isinstance(description, str) else str(description)
        )

    merged: list[dict[str, str]] = []

    for column in dataframe.columns:
        name = str(column)
        merged.append(
            {
                "name": name,
                "type": _pandas_dtype_to_sql_type(dataframe[column].dtype),
                "description": descriptions.get(name, ""),
            }
        )

    return merged


def _analysis_text(
    table_name: str,
    workbook_description: str,
    analysis: dict[str, Any],
    context: str,
) -> str:
    """
    Build searchable text containing workbook meaning, schema, relationships,
    surrounding text context, and representative table values.
    """

    parts = [f"Table: {table_name}"]

    if workbook_description:
        parts.extend(("Workbook context:", workbook_description))

    if context:
        parts.extend(("Related document text:", context))

    if analysis["role"]:
        parts.append(f"Role: {analysis['role']}")

    if analysis["description"]:
        parts.extend(("Description:", analysis["description"]))

    if analysis["schema"]:
        parts.append("Schema:")

        for column in analysis["schema"]:
            if isinstance(column, dict):
                name = column.get("name", "Unknown column")
                sql_type = column.get("type", "TEXT")
                description = column.get("description", "")
                line = (f"- {name} " f"({sql_type}): " f"{description}").rstrip()
                parts.append(line)
            else:
                parts.append(f"- {column}")

    if analysis["relationships"]:
        parts.append("Related tables:")

        for relationship in analysis["relationships"]:
            if isinstance(relationship, dict):
                target = (
                    relationship.get("sheet_name")
                    or f"table " f"{relationship.get('table_index', 'unknown')}"
                )
                description = relationship.get("relationship", "")
                parts.append(f"- {target}: " f"{description}")
            else:
                parts.append(f"- {relationship}")

    return "\n".join(parts)


def _add_related_table_names(
    table_analyses: list[dict[str, Any]],
    tables: list[ExtractedTable],
) -> None:
    """Convert model-generated table indexes into stable table names."""

    for analysis in table_analyses:
        enriched_relationships: list[Any] = []

        for relationship in analysis["relationships"]:
            if not isinstance(
                relationship,
                dict,
            ):
                enriched_relationships.append(relationship)
                continue

            enriched = relationship.copy()

            index = enriched.get("table_index")

            if isinstance(index, int) and 0 <= index < len(tables):
                enriched["sheet_name"] = tables[index].name

            enriched_relationships.append(enriched)

        analysis["relationships"] = enriched_relationships


async def process_table(payload: ProcessorPayload) -> list[Document]:
    """Create chunks enriched by a single workbook-level LLM analysis."""

    tables = await asyncio.to_thread(
        _extract_tables,
        payload,
    )

    if not tables:
        return []

    # Collect surrounding document text once.
    context = _text_context(payload.elements)

    # Build a compact catalog from DataFrames.
    catalog = _build_catalog(tables)

    # LLM generates semantic descriptions only.
    # Pandas remains authoritative for data types.
    workbook_description = ""
    table_analyses = [_empty_analysis() for _ in tables]

    try:
        response = await payload.llm.answer(
            WORKBOOK_ANALYSIS_PROMPT.format(
                context=context or "(none)",
                catalog=catalog,
            )
        )

        workbook_description, table_analyses = _parse_workbook_analysis(
            response,
            len(tables),
        )

    except Exception:
        # The raw DataFrame data and pandas-derived schema are still indexed
        # even if the LLM fails.
        pass

    _add_related_table_names(table_analyses, tables)

    chunks: list[Document] = []

    for index, table in enumerate(tables):
        analysis = table_analyses[index]

        # Pandas is the source of truth for schema types.
        analysis["schema"] = _merge_schema_types(table.dataframe, analysis["schema"])

        table_name = table.name

        table_rows = table.dataframe.to_dict(orient="records")

        chunks.append(
            Document(
                category=DocumentCategory.SPREADSHEET,
                text=_analysis_text(
                    table_name=table_name,
                    workbook_description=workbook_description,
                    analysis=analysis,
                    context=context,
                ),
                metadata={
                    # reference
                    "chunk_source_id": payload.source_id,
                    "chunk_source_page_number": table.page_number,
                    # sql data
                    "sql_table_name": table_name,
                    "sql_schema": analysis["schema"],
                    "sql_rows": table_rows,
                },
                orig_elements=table.orig_elements,
            )
        )

    return chunks
