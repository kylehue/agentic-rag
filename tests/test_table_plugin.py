import asyncio
import io
import json

import pandas as pd

from app.plugin.context import IngestionContext
from app.plugin.hooks import IngestionProcessPayload
from app.plugin.registry import PluginRegistry

from fakes import FakeLLM, build_runtime, make_context
from app.plugins.table import TablePlugin

CSV_BYTES = b"Region,Amount\nnorth,10\nsouth,20\neast,30\n"


def make_analysis():
    return {
        "workbook_description": "Regional sales figures.",
        "tables": [
            {
                "index": 0,
                "description": "Sales per region.",
                "role": "fact data",
                "schema": [
                    {"name": "Region", "description": "sales region"},
                    {"name": "Amount", "description": "total amount"},
                ],
                "relationships": [],
            }
        ],
    }


def csv_context(
    filename: str = "sales.csv",
    content_type: str = "text/csv",
    source_bytes: bytes = CSV_BYTES,
    parent_source_id: str | None = None,
    source_description: str | None = None,
) -> IngestionContext:
    return make_context(
        filename=filename,
        content_type=content_type,
        source_bytes=source_bytes,
        parent_source_id=parent_source_id,
        source_description=source_description,
    )


def run_process(context, runtime, plugin: TablePlugin):
    """Register the plugin on the runtime's bus and fire the process hook."""
    PluginRegistry(runtime.hooks).register(plugin)

    async def flow():
        results = await runtime.hooks.trigger(
            "ingestion_process",
            IngestionProcessPayload(context=context, runtime=runtime),
        )
        chunks = []
        for result in results:
            if result:
                chunks.extend(result)
        return chunks

    return asyncio.run(flow())


def test_process_csv_returns_chunk_preserving_the_original_schema():
    llm = FakeLLM(json.dumps(make_analysis()))
    plugin = TablePlugin()
    context = csv_context()
    parts = build_runtime(context, llm=llm)

    chunks = run_process(context, parts.runtime, plugin)

    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.plugin == "table"

    # Searchable text carries the LLM analysis and a data preview.
    assert "Table: sales" in chunk.text
    assert "Regional sales figures." in chunk.text
    assert "- Region (TEXT): sales region" in chunk.text
    assert "Data:" in chunk.text
    assert "north" in chunk.text

    # The original file schema is preserved: column names are kept verbatim
    # (not normalized/safe-stringed), types come from pandas.
    assert chunk.metadata["table_name"] == "sales"
    assert chunk.metadata["schema"] == [
        {"name": "Region", "type": "TEXT", "description": "sales region"},
        {"name": "Amount", "type": "INTEGER", "description": "total amount"},
    ]

    # No rows are loaded into SQL and no file is emitted: the chunk is
    # text + metadata only.
    assert not any(key.startswith("sql_") for key in chunk.metadata)
    assert parts.runtime.pop_emitted_files() == []

    # A single workbook-level LLM call.
    assert len(llm.prompts) == 1


def test_process_xlsx_returns_chunk_per_sheet():
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer) as writer:  # type: ignore[reportArgumentType]
        pd.DataFrame({"a": [1, 2]}).to_excel(
            writer,
            sheet_name="Alpha",
            index=False,
        )
        pd.DataFrame({"b": [3.5]}).to_excel(
            writer,
            sheet_name="Beta 2",
            index=False,
        )

    llm = FakeLLM(
        json.dumps(
            {
                "workbook_description": "Two sheets.",
                "tables": [
                    {
                        "index": 0,
                        "description": "alpha data",
                        "role": "",
                        "schema": [],
                        "relationships": [],
                    },
                    {
                        "index": 1,
                        "description": "beta data",
                        "role": "",
                        "schema": [],
                        "relationships": [
                            {"table_index": 0, "relationship": "joins alpha"}
                        ],
                    },
                ],
            }
        )
    )
    plugin = TablePlugin()
    context = csv_context(
        filename="book.xlsx",
        content_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        source_bytes=buffer.getvalue(),
    )
    parts = build_runtime(context, llm=llm)

    chunks = run_process(context, parts.runtime, plugin)

    assert len(chunks) == 2
    # Sheet names are preserved verbatim from the workbook.
    names = {chunk.metadata["table_name"] for chunk in chunks}
    assert names == {"Alpha", "Beta 2"}

    # Relationship indexes are rewritten to stable table names.
    beta = next(chunk for chunk in chunks if chunk.metadata["table_name"] == "Beta 2")
    assert "Alpha" in beta.text
    assert "joins alpha" in beta.text

    # No files are emitted: the table plugin produces text + metadata only.
    assert parts.runtime.pop_emitted_files() == []


def test_llm_failure_falls_back_to_raw_data():
    llm = FakeLLM("this is not json")
    plugin = TablePlugin()
    context = csv_context()
    parts = build_runtime(context, llm=llm)

    chunks = run_process(context, parts.runtime, plugin)

    assert len(chunks) == 1
    assert "Table: sales" in chunks[0].text
    assert "Data:" in chunks[0].text
    assert "Region" in chunks[0].text


def test_source_description_is_passed_to_the_llm():
    llm = FakeLLM(json.dumps(make_analysis()))
    plugin = TablePlugin()
    context = csv_context(
        source_description="This table was extracted from a PDF about Q3 sales.",
    )
    parts = build_runtime(context, llm=llm)

    run_process(context, parts.runtime, plugin)

    assert len(llm.prompts) == 1
    assert "This table was extracted from a PDF about Q3 sales." in llm.prompts[0]


def test_no_source_description_leaves_prompt_unchanged():
    llm = FakeLLM(json.dumps(make_analysis()))
    plugin = TablePlugin()
    context = csv_context()
    parts = build_runtime(context, llm=llm)

    run_process(context, parts.runtime, plugin)

    assert len(llm.prompts) == 1
    assert "Additional context about the source document" not in llm.prompts[0]


def test_blank_source_description_is_ignored():
    llm = FakeLLM(json.dumps(make_analysis()))
    plugin = TablePlugin()
    context = csv_context(source_description="   ")
    parts = build_runtime(context, llm=llm)

    run_process(context, parts.runtime, plugin)

    assert "Additional context about the source document" not in llm.prompts[0]


def test_rejects_unsupported_spreadsheet_extension():
    llm = FakeLLM("")
    plugin = TablePlugin()
    context = csv_context(filename="data.tsv", source_bytes=b"a\tb\n1\t2")
    parts = build_runtime(context, llm=llm)

    chunks = run_process(context, parts.runtime, plugin)

    assert chunks == []
    assert llm.prompts == []
