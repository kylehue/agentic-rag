import json
from typing import Any

from bs4 import BeautifulSoup
from app.dependencies import llm
from app.models.document import Document, DocumentProcessorChunk
from unstructured.documents.elements import Element, Table

MAX_WORKBOOK_CONTEXT_CHARS = 30_000
MAX_TABLE_CONTEXT_CHARS = 10_000
SAMPLED_DATA_ROWS_PER_TABLE = 3

WORKBOOK_ANALYSIS_PROMPT = """You are analyzing a workbook for a retrieval system.
The input is a catalog of tables extracted from one workbook. Tables may refer to,
define, summarize, or join other tables. Analyze the catalog as a whole, so each
table's meaning is informed by the other sheets.

Return valid JSON only in this exact shape:
{
  "workbook_description": "one concise description of the workbook",
  "tables": [
    {
      "index": 0,
      "description": "what this table contains and its role in the workbook",
      "role": "for example: lookup, fact data, summary, instructions, assumptions",
      "schema": [
        {"name": "column name", "type": "inferred data type", "description": "meaning"}
      ],
      "relationships": [
        {"table_index": 1, "relationship": "how this table relates to that table"}
      ]
    }
  ]
}

Include every catalog index exactly once. Use an empty relationships list when no
relationship is supported by the data. Do not invent joins, formulas, or facts.

Workbook catalog:
{catalog}
"""


def _table_label(table: Table, index: int) -> str:
    """Use the XLSX worksheet name emitted by Unstructured when available."""
    return getattr(table.metadata, "page_name", None) or f"Table {index + 1}"


def _source_rows(table: Table) -> list[str]:
    """Return readable rows, preferring the table structure retained by Unstructured."""
    html = getattr(table.metadata, "text_as_html", None)
    if not html:
        return [line for line in (table.text or "").splitlines() if line.strip()]

    soup = BeautifulSoup(html, "html.parser")
    rows = []
    for row in soup.find_all("tr"):
        cells = [cell.get_text(" ", strip=True) for cell in row.find_all(["th", "td"])]
        if cells:
            rows.append(" | ".join(cells))
    return rows or [line for line in (table.text or "").splitlines() if line.strip()]


def _sample_table_rows(table: Table, char_limit: int) -> str:
    """Give the LLM headers and representative data, never an entire large sheet."""
    rows = _source_rows(table)
    if not rows:
        return "[empty table]"

    header, data_rows = rows[0], rows[1:]
    if len(data_rows) <= SAMPLED_DATA_ROWS_PER_TABLE:
        sampled_rows = data_rows
        omitted_count = 0
    else:
        head_count = SAMPLED_DATA_ROWS_PER_TABLE // 2
        tail_count = SAMPLED_DATA_ROWS_PER_TABLE - head_count
        sampled_rows = data_rows[:head_count] + data_rows[-tail_count:]
        omitted_count = len(data_rows) - len(sampled_rows)

    preview_lines = [header, *sampled_rows]
    if omitted_count:
        preview_lines.insert(
            1 + SAMPLED_DATA_ROWS_PER_TABLE // 2,
            f"[{omitted_count:,} data rows omitted from LLM context]",
        )
    preview = "\n".join(preview_lines)
    if len(preview) > char_limit:
        return preview[:char_limit] + "\n[preview truncated]"
    return preview


def _build_catalog(tables: list[Table]) -> str:
    """Create a balanced, row-sampled preview so no large sheet hides the rest."""
    per_table_limit = min(
        MAX_TABLE_CONTEXT_CHARS,
        max(1, MAX_WORKBOOK_CONTEXT_CHARS // len(tables)),
    )
    catalog = []
    for i, table in enumerate(tables):
        catalog.append(
            {
                "index": i,
                "sheet_name": _table_label(table, i),
                "table_id": getattr(table.metadata, "table_id", None),
                "data_preview": _sample_table_rows(table, per_table_limit),
            }
        )
    return json.dumps(catalog, ensure_ascii=False)


def _parse_json(response: str) -> Any:
    """Accept JSON wrapped in a Markdown code fence, as models occasionally do."""
    response = response.strip()
    if response.startswith("```"):
        response = response.split("\n", 1)[-1]
        if response.rstrip().endswith("```"):
            response = response.rstrip()[:-3].rstrip()
    return json.loads(response)


def _empty_analysis() -> dict[str, Any]:
    return {"description": "", "role": "", "schema": [], "relationships": []}


def _parse_workbook_analysis(
    response: str, table_count: int
) -> tuple[str, list[dict[str, Any]]]:
    """Normalize model output and retain an empty analysis for malformed entries."""
    payload = _parse_json(response)
    if not isinstance(payload, dict):
        raise ValueError("Workbook analysis must be a JSON object")

    workbook_description = payload.get("workbook_description", "")
    if not isinstance(workbook_description, str):
        workbook_description = str(workbook_description)

    analyses = [_empty_analysis() for _ in range(table_count)]
    table_analyses = payload.get("tables", [])
    if not isinstance(table_analyses, list):
        return workbook_description, analyses

    for item in table_analyses:
        if not isinstance(item, dict) or not isinstance(item.get("index"), int):
            continue
        index = item["index"]
        if not 0 <= index < table_count:
            continue
        analyses[index] = {
            "description": (
                item.get("description", "")
                if isinstance(item.get("description", ""), str)
                else str(item.get("description", ""))
            ),
            "role": (
                item.get("role", "")
                if isinstance(item.get("role", ""), str)
                else str(item.get("role", ""))
            ),
            "schema": (
                item.get("schema", [])
                if isinstance(item.get("schema", []), list)
                else []
            ),
            "relationships": (
                item.get("relationships", [])
                if isinstance(item.get("relationships", []), list)
                else []
            ),
        }
    return workbook_description, analyses


def _analysis_text(
    sheet_name: str,
    workbook_description: str,
    analysis: dict[str, Any],
    table_text: str,
) -> str:
    """Make workbook context, schema, links, and original values searchable together."""
    parts = [f"Workbook table: {sheet_name}"]
    if workbook_description:
        parts.extend(("Workbook context:", workbook_description))
    if analysis["role"]:
        parts.append(f"Role: {analysis['role']}")
    if analysis["description"]:
        parts.extend(("Description:", analysis["description"]))
    if analysis["schema"]:
        parts.append("Schema:")
        for column in analysis["schema"]:
            if isinstance(column, dict):
                parts.append(
                    f"- {column.get('name', 'Unknown column')} ({column.get('type', 'unknown')}): "
                    f"{column.get('description', '')}".rstrip()
                )
            else:
                parts.append(f"- {column}")
    if analysis["relationships"]:
        parts.append("Related tables:")
        for relationship in analysis["relationships"]:
            if isinstance(relationship, dict):
                target = (
                    relationship.get("sheet_name")
                    or f"table {relationship.get('table_index', 'unknown')}"
                )
                parts.append(
                    f"- {target}: {relationship.get('relationship', '')}".rstrip()
                )
            else:
                parts.append(f"- {relationship}")
    parts.extend(("Data:", table_text))
    return "\n".join(parts)


def _add_related_sheet_names(
    analyses: list[dict[str, Any]], tables: list[Table]
) -> None:
    """Turn model indexes into stable, human-readable worksheet references."""
    for analysis in analyses:
        enriched_relationships = []
        for relationship in analysis["relationships"]:
            if not isinstance(relationship, dict):
                enriched_relationships.append(relationship)
                continue
            enriched = relationship.copy()
            index = enriched.get("table_index")
            if isinstance(index, int) and 0 <= index < len(tables):
                enriched["sheet_name"] = _table_label(tables[index], index)
            enriched_relationships.append(enriched)
        analysis["relationships"] = enriched_relationships


async def process_table(
    document: Document,
    elements: list[Element],
) -> list[DocumentProcessorChunk]:
    """Create chunks enriched by a single workbook-level LLM analysis."""
    tables = [e for e in elements if isinstance(e, Table)]
    if not tables:
        return []

    workbook_description = ""
    analyses = [_empty_analysis() for _ in tables]
    try:
        response = await llm.answer(
            WORKBOOK_ANALYSIS_PROMPT.replace("{catalog}", _build_catalog(tables))
        )
        workbook_description, analyses = _parse_workbook_analysis(response, len(tables))
    except Exception:
        # Index source data even when the provider is unavailable or malformed.
        pass

    _add_related_sheet_names(analyses, tables)

    chunks: list[DocumentProcessorChunk] = []
    for i, table in enumerate(tables):
        analysis = analyses[i]
        sheet_name = _table_label(table, i)
        chunks.append(
            DocumentProcessorChunk(
                id=f"{document.id}:table:{i}",
                document=document,
                text=_analysis_text(
                    sheet_name, workbook_description, analysis, table.text
                ),
                orig_elements=[table],
                metadata={
                    "page_number": getattr(table.metadata, "page_number", None),
                    "sheet_name": sheet_name,
                    "table_html": getattr(table.metadata, "text_as_html", None),
                    "workbook_description": workbook_description,
                    "description": analysis["description"],
                    "role": analysis["role"],
                    "schema": analysis["schema"],
                    "relationships": analysis["relationships"],
                },
            )
        )
    return chunks
