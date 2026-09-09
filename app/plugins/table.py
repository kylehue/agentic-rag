from __future__ import annotations

import asyncio
from dataclasses import dataclass
from io import BytesIO
import json
from pathlib import Path
from typing import Any

import pandas as pd

from app.llm.base import LLMProvider
from app.models.chunk import IngestedChunk
from app.plugin.base import Plugin
from app.plugin.context import IngestionContext
from app.plugin.runtime import IngestionRuntime
from app.utils.string import render_template

MAX_WORKBOOK_CONTEXT_CHARS = 30_000
MAX_TABLE_CONTEXT_CHARS = 10_000
SAMPLED_DATA_ROWS_PER_TABLE = 5


WORKBOOK_ANALYSIS_PROMPT_TEMPLATE = """You are describing a set of tables extracted from one source for a retrieval system.

The input is a catalog of tables. Describe the source and each table well
enough that a searcher can find them: what they contain, what the columns
mean, the granularity of the rows, and the kinds of questions the tables can
answer.

Return valid JSON only in this exact shape:
{
  "workbook_description": "One concise description of the whole source optimized for RAG retrieval. Prioritize keyword density and explicit context over human readability. Just mention the possible use-cases or search queries this source answers, no need to describe the schema.",
  "tables": [
    {
      "index": 0,
      "description": "What this table contains, what its columns mean, its row granularity, and the search queries it can answer. Optimized for RAG retrieval: keyword-dense and explicit over human readability."
    }
  ]
}

Include every catalog index exactly once.
Do not invent facts that are not supported by the data.

{file_description}
Table catalog:
{catalog}
"""


@dataclass(frozen=True)
class ExtractedTable:
    """A table represented independently of its source format."""

    dataframe: pd.DataFrame
    name: str


class TablePlugin(Plugin):
    """Handles spreadsheet documents (csv, xlsx, xls)."""

    SUPPORTED_EXTENSIONS = frozenset({"csv", "xlsx", "xls"})

    @property
    def name(self) -> str:
        return "table"

    def accepts(self, context: IngestionContext) -> bool:
        extension = Path(context.file.filename).suffix.lower().lstrip(".")
        return extension in self.SUPPORTED_EXTENSIONS

    async def on_ingestion_process(
        self,
        context: IngestionContext,
        runtime: IngestionRuntime,
    ) -> list[IngestedChunk]:
        """Generate one chunk per table, enriched by a workbook-level LLM description."""
        if not self.accepts(context):
            return []

        tables = await asyncio.to_thread(
            self._read_tables,
            context,
        )

        if not tables:
            return []

        # Build a compact catalog from DataFrames.
        catalog = self._build_catalog(tables)

        # The LLM generates retrieval-optimized descriptions only.
        # Schema and relationship discovery is left to the agent that
        # queries the stored tables.
        workbook_description, table_analyses = await self._analyze_workbook(
            runtime.llm,
            catalog,
            len(tables),
            context.file.description,
        )

        chunks: list[IngestedChunk] = []

        for index, table in enumerate(tables):
            analysis = table_analyses[index]

            chunks.append(
                IngestedChunk(
                    plugin=self.name,
                    text=self._analysis_text(
                        table_name=table.name,
                        workbook_description=workbook_description,
                        description=analysis["description"],
                        dataframe=table.dataframe,
                    ),
                    metadata={"table_name": table.name},
                )
            )

        return chunks

    @staticmethod
    def _read_tables(
        context: IngestionContext,
    ) -> list[ExtractedTable]:
        """
        Read spreadsheet data directly with pandas, keeping the schema as-is.

        Column names and sheet names come straight from the file: they are not
        normalized, deduplicated, or rewritten into safe identifiers.

        CSV produces one table named after the file.
        XLS/XLSX produces one table per sheet, named after the sheet.
        """

        extension = Path(context.file.filename).suffix.lower()

        source = BytesIO(context.file.file_bytes)

        if extension == ".csv":
            return [
                ExtractedTable(
                    dataframe=pd.read_csv(source),
                    name=Path(context.file.filename).stem,
                )
            ]

        if extension in {".xlsx", ".xls"}:
            sheets = pd.read_excel(
                source,
                sheet_name=None,
            )

            return [
                ExtractedTable(
                    dataframe=dataframe,
                    name=str(sheet_name),
                )
                for sheet_name, dataframe in sheets.items()
            ]

        return []

    async def _analyze_workbook(
        self,
        llm: LLMProvider,
        catalog: str,
        table_count: int,
        file_description: str | None = None,
    ) -> tuple[str, list[dict[str, Any]]]:
        """Generate the single workbook-level LLM analysis.

        ``file_description`` is optional context about the source document
        (often the text found around these tables in the parent document).
        When present it is given to the LLM to inform the descriptions.

        Returns a safe fallback when the LLM fails or answers invalidly;
        the table's sample data is still indexed.
        """

        try:
            prompt = render_template(
                WORKBOOK_ANALYSIS_PROMPT_TEMPLATE,
                {
                    "catalog": catalog,
                    "file_description": self._file_description_block(file_description),
                },
            )

            response = await llm.answer(prompt)

            return self._parse_workbook_analysis(
                response,
                table_count,
            )

        except Exception:
            return "", [self._empty_analysis() for _ in range(table_count)]

    @staticmethod
    def _file_description_block(file_description: str | None) -> str:
        """Render the optional file description as a prompt block ("" if absent)."""
        if not file_description or not file_description.strip():
            return ""

        return (
            "Additional context about the source document, provided by the "
            "caller:\n"
            f"{file_description.strip()}\n\n"
            "Use this context to inform your descriptions.\n\n"
        )

    @staticmethod
    def _pandas_dtype_to_type(
        dtype: Any,
    ) -> str:
        """
        Convert a pandas dtype into a descriptive column type.

        The types are intentionally conservative because the DataFrame is the
        source of truth for the data.
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

    @staticmethod
    def _schema_for_prompt(
        dataframe: pd.DataFrame,
    ) -> list[dict[str, str]]:
        """
        Build the schema representation sent to the LLM.

        The LLM uses the column names and inferred types to ground the
        descriptions it writes.
        """

        return [
            {
                "name": str(column),
                "type": TablePlugin._pandas_dtype_to_type(dataframe[column].dtype),
                "description": "",
            }
            for column in dataframe.columns
        ]

    @staticmethod
    def _sample_table_rows(
        dataframe: pd.DataFrame,
        char_limit: int,
    ) -> str:
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

    @classmethod
    def _build_catalog(
        cls,
        tables: list[ExtractedTable],
    ) -> str:
        """
        Create a row-sampled catalog containing each table's schema and data preview.
        """

        if not tables:
            return "[]"

        per_table_limit = min(
            MAX_TABLE_CONTEXT_CHARS,
            max(
                1,
                MAX_WORKBOOK_CONTEXT_CHARS // len(tables),
            ),
        )

        catalog: list[dict[str, Any]] = []

        for index, table in enumerate(tables):
            catalog.append(
                {
                    "index": index,
                    "table_name": table.name,
                    "row_count": len(table.dataframe),
                    "column_count": len(table.dataframe.columns),
                    "schema": cls._schema_for_prompt(table.dataframe),
                    "data_preview": cls._sample_table_rows(
                        table.dataframe,
                        per_table_limit,
                    ),
                }
            )

        return json.dumps(
            catalog,
            ensure_ascii=False,
        )

    @staticmethod
    def _parse_json(
        response: str,
    ) -> Any:
        """Accept JSON wrapped in a Markdown code fence."""

        response = response.strip()

        if response.startswith("```"):
            response = response.split(
                "\n",
                1,
            )[-1]

            if response.rstrip().endswith("```"):
                response = response.rstrip()[:-3].rstrip()

        return json.loads(response)

    @staticmethod
    def _empty_analysis() -> dict[str, Any]:
        return {"description": ""}

    @classmethod
    def _parse_workbook_analysis(
        cls,
        response: str,
        table_count: int,
    ) -> tuple[str, list[dict[str, Any]]]:
        """Normalize model output and preserve missing table analyses."""

        payload = cls._parse_json(response)

        if not isinstance(payload, dict):
            raise ValueError("Workbook analysis must be a JSON object.")

        workbook_description = payload.get("workbook_description", "")

        if not isinstance(workbook_description, str):
            workbook_description = str(workbook_description)

        analyses = [cls._empty_analysis() for _ in range(table_count)]

        table_analyses = payload.get("tables", [])

        if not isinstance(table_analyses, list):
            return (workbook_description, analyses)

        for item in table_analyses:
            if not isinstance(item, dict):
                continue

            index = item.get("index")

            if not isinstance(index, int):
                continue

            if not 0 <= index < table_count:
                continue

            description = item.get("description", "")

            analyses[index] = {
                "description": (
                    description if isinstance(description, str) else str(description)
                ),
            }

        return (workbook_description, analyses)

    @staticmethod
    def _analysis_text(
        table_name: str,
        workbook_description: str,
        description: str,
        dataframe: pd.DataFrame,
    ) -> str:
        """
        Build searchable text: the table's identity, the workbook context, the
        retrieval-optimized description, and representative values.
        """

        parts = [f"Table: {table_name}"]

        if workbook_description:
            parts.extend(("Workbook context:", workbook_description))

        if description:
            parts.extend(("Description:", description))

        parts.extend(
            (
                "Sample Data:",
                TablePlugin._sample_table_rows(
                    dataframe,
                    MAX_TABLE_CONTEXT_CHARS,
                ),
            )
        )

        return "\n".join(parts)
