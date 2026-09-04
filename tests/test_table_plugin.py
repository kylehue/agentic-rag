import asyncio
import io
import json

import pandas as pd

from app.core.config import settings

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
                    {"name": "region", "description": "sales region"},
                    {"name": "amount", "description": "total amount"},
                ],
                "relationships": [],
            }
        ],
    }


def csv_context(**kwargs):
    defaults = dict(
        filename="sales.csv",
        content_type="text/csv",
        source_bytes=CSV_BYTES,
    )
    defaults.update(kwargs)
    return make_context(**defaults)  # type: ignore


def test_process_csv_creates_one_chunk_with_file():
    llm = FakeLLM(json.dumps(make_analysis()))
    plugin = TablePlugin()
    context = csv_context()
    parts = build_runtime(context, llm=llm)

    chunks = asyncio.run(plugin.process(context, parts.runtime))

    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.plugin == "table"

    # Searchable text carries the LLM analysis and a data preview.
    assert "Table: sales" in chunk.text
    assert "Regional sales figures." in chunk.text
    assert "- region (TEXT): sales region" in chunk.text
    assert "Data:" in chunk.text
    assert "north" in chunk.text

    # No rows are loaded into SQL: no sql_* metadata keys.
    assert not any(key.startswith("sql_") for key in chunk.metadata)

    # The chunk carries its own CSV file.
    assert chunk.file_filename == "sales.csv"
    assert chunk.file_content_type == "text/csv"
    assert chunk.file_bytes is not None
    assert b"north,10" in chunk.file_bytes

    # The plugin persisted the chunk through the runtime.
    chunk_rows = parts.sql_storage.tables[settings.CHUNK_TABLE_NAME]
    assert len(chunk_rows) == 1
    metadata = chunk_rows[0]["metadata"]
    assert metadata["chunk_source_id"] == context.source_id
    assert metadata["chunk_table_name"] == "sales"
    assert metadata["chunk_schema"] == [
        {"name": "region", "type": "TEXT", "description": "sales region"},
        {"name": "amount", "type": "INTEGER", "description": "total amount"},
    ]
    assert metadata["chunk_file_path"].endswith(".csv")
    assert metadata["chunk_file_content_type"] == "text/csv"

    # The chunk file landed in file storage.
    chunk_file_paths = [
        path for path in parts.file_storage.files if path.startswith("chunk_files/")
    ]
    assert len(chunk_file_paths) == 1

    # Vectors were stored.
    assert parts.vector_storage.added[0][0] == [chunk.chunk_id]

    # A single workbook-level LLM call.
    assert len(llm.prompts) == 1


def test_process_xlsx_creates_chunk_per_sheet():
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer) as writer:
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

    chunks = asyncio.run(plugin.process(context, parts.runtime))

    assert len(chunks) == 2
    names = {chunk.metadata["chunk_table_name"] for chunk in chunks}
    assert names == {"alpha", "beta_2"}

    # Relationship indexes are rewritten to stable table names.
    beta = next(
        chunk for chunk in chunks if chunk.metadata["chunk_table_name"] == "beta_2"
    )
    assert "alpha" in beta.text
    assert "joins alpha" in beta.text

    # Each chunk carries its own CSV file.
    for chunk in chunks:
        assert chunk.file_bytes is not None
        assert chunk.file_filename.endswith(".csv")  # type: ignore


def test_llm_failure_falls_back_to_raw_data():
    llm = FakeLLM("this is not json")
    plugin = TablePlugin()
    context = csv_context()
    parts = build_runtime(context, llm=llm)

    chunks = asyncio.run(plugin.process(context, parts.runtime))

    assert len(chunks) == 1
    assert "Table: sales" in chunks[0].text
    assert "Data:" in chunks[0].text
    assert "region" in chunks[0].text


def test_parent_source_id_recorded():
    llm = FakeLLM(json.dumps(make_analysis()))
    plugin = TablePlugin()
    context = csv_context(parent_source_id="parent-1")
    parts = build_runtime(context, llm=llm)

    chunks = asyncio.run(plugin.process(context, parts.runtime))

    chunk_rows = parts.sql_storage.tables[settings.CHUNK_TABLE_NAME]
    assert chunk_rows[0]["metadata"]["chunk_parent_source_id"] == "parent-1"


def test_unsupported_spreadsheet_extension_returns_no_chunks():
    llm = FakeLLM("")
    plugin = TablePlugin()
    context = csv_context(filename="data.tsv", source_bytes=b"a\tb\n1\t2")
    parts = build_runtime(context, llm=llm)

    chunks = asyncio.run(plugin.process(context, parts.runtime))

    assert chunks == []
    assert llm.prompts == []
