import asyncio
import io
import json

import pandas as pd

from app.plugin.context import IngestionContext
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


def run_process(context, plugin: TablePlugin, **runtime_kwargs):
    """Wire the plugin through a registry and run the ingestion process.

    Returns (chunks, parts); parts carries the runtime and the fakes.
    """
    registry = PluginRegistry()
    registry.register(plugin)
    parts = build_runtime(context, registry=registry, **runtime_kwargs)

    async def flow():
        return await registry.ingestion_process(context, parts.runtime)

    return asyncio.run(flow()), parts


def test_process_csv_returns_a_retrieval_chunk():
    llm = FakeLLM(json.dumps(make_analysis()))
    context = csv_context()

    chunks, parts = run_process(context, TablePlugin(), llm=llm)

    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.plugin == "table"

    # Searchable text carries the workbook context, the LLM description,
    # and a data preview.
    assert "Table: sales" in chunk.text
    assert "Regional sales figures." in chunk.text
    assert "Sales per region." in chunk.text
    assert "Sample Data:" in chunk.text
    assert "north" in chunk.text

    # Metadata carries the name plus the precomputed schema, so the agent can
    # inspect and query the table without reading the file. The description is
    # deliberately not duplicated here. It is already in the chunk's text.
    assert chunk.metadata == {
        "table_name": "sales",
        "schema": [
            {"name": "Region", "type": "TEXT"},
            {"name": "Amount", "type": "INTEGER"},
        ],
        "row_count": 3,
        "column_count": 2,
    }

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
                    {"index": 0, "description": "alpha data"},
                    {"index": 1, "description": "beta data"},
                ],
            }
        )
    )
    context = csv_context(
        filename="book.xlsx",
        content_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        source_bytes=buffer.getvalue(),
    )

    chunks, parts = run_process(context, TablePlugin(), llm=llm)

    assert len(chunks) == 2
    # Sheet names are preserved verbatim from the workbook.
    names = {chunk.metadata["table_name"] for chunk in chunks}
    assert names == {"Alpha", "Beta 2"}

    # The chunk text carries the workbook context and the table's description.
    beta = next(chunk for chunk in chunks if chunk.metadata["table_name"] == "Beta 2")
    assert "Two sheets." in beta.text
    assert "beta data" in beta.text

    # No files are emitted: the table plugin produces text + metadata only.
    assert parts.runtime.pop_emitted_files() == []


def test_llm_failure_falls_back_to_raw_data():
    llm = FakeLLM("this is not json")
    context = csv_context()

    chunks, parts = run_process(context, TablePlugin(), llm=llm)

    assert len(chunks) == 1
    assert "Table: sales" in chunks[0].text
    # The description is empty on fallback, but the sample data is indexed.
    assert "Description:" not in chunks[0].text
    assert "Sample Data:" in chunks[0].text
    assert "Region" in chunks[0].text


def test_source_description_is_passed_to_the_llm():
    llm = FakeLLM(json.dumps(make_analysis()))
    context = csv_context(
        source_description="This table was extracted from a PDF about Q3 sales.",
    )

    run_process(context, TablePlugin(), llm=llm)

    assert len(llm.prompts) == 1
    assert "This table was extracted from a PDF about Q3 sales." in llm.prompts[0]


def test_no_source_description_leaves_prompt_unchanged():
    llm = FakeLLM(json.dumps(make_analysis()))
    context = csv_context()

    run_process(context, TablePlugin(), llm=llm)

    assert len(llm.prompts) == 1
    assert "Additional context about the source document" not in llm.prompts[0]


def test_blank_source_description_is_ignored():
    llm = FakeLLM(json.dumps(make_analysis()))
    context = csv_context(source_description="   ")

    run_process(context, TablePlugin(), llm=llm)

    assert "Additional context about the source document" not in llm.prompts[0]


def test_rejects_unsupported_spreadsheet_extension():
    llm = FakeLLM("")
    context = csv_context(filename="data.tsv", source_bytes=b"a\tb\n1\t2")

    chunks, parts = run_process(context, TablePlugin(), llm=llm)

    assert chunks == []
    assert llm.prompts == []
