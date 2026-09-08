import asyncio
from io import BytesIO
from unittest.mock import patch

import pandas as pd

from app.plugin.hooks import IngestionProcessPayload
from app.plugin.registry import PluginRegistry

from fakes import (
    build_runtime,
    make_context,
    make_footer_element,
    make_image_element,
    make_page_break_element,
    make_table_element,
    make_text_element,
)
from app.plugins import text as text_module
from app.plugins.text import TextPlugin

# name | age table with header
TABLE_A = (
    "<table><thead><tr><th>name</th><th>age</th></tr></thead>"
    "<tbody><tr><td>kyle</td><td>25</td></tr>"
    "<tr><td>anne</td><td>21</td></tr></tbody></table>"
)

# continuation page: repeated header + one row
TABLE_A_REPEAT_HEADER = (
    "<table><thead><tr><th>name</th><th>age</th></tr></thead>"
    "<tbody><tr><td>john</td><td>23</td></tr></tbody></table>"
)

# continuation page: no header, numeric age values (same schema as TABLE_A)
TABLE_A_NO_HEADER = (
    "<table><tbody><tr><td>john</td><td>44</td></tr>"
    "<tr><td>anika</td><td>16</td></tr></tbody></table>"
)

# different schema: part | length with non-numeric values
TABLE_B = (
    "<table><thead><tr><th>part</th><th>length</th></tr></thead>"
    "<tbody><tr><td>petal</td><td>5cm</td></tr>"
    "<tr><td>sepal</td><td>2cm</td></tr></tbody></table>"
)

# headerless with non-numeric second column (conflicts with TABLE_A's age)
TABLE_B_NO_HEADER = (
    "<table><tbody><tr><td>petal</td><td>4cm</td></tr>"
    "<tr><td>sepal</td><td>1cm</td></tr></tbody></table>"
)


def run_process(context, plugin: TextPlugin, partition_result, **runtime_kwargs):
    """Wire the plugin through a registry and fire the process hook.

    Returns (chunks, parts); parts carries the runtime and the fakes.
    """
    registry = PluginRegistry()
    registry.register(plugin)
    parts = build_runtime(context, hooks=registry.hooks, **runtime_kwargs)

    async def flow():
        with patch.object(
            text_module,
            "partition",
            return_value=partition_result,
        ):
            results = await parts.runtime.hooks.trigger(
                "ingestion_process",
                IngestionProcessPayload(context=context, runtime=parts.runtime),
            )
        chunks = []
        for result in results:
            if result:
                chunks.extend(result)
        return chunks

    return asyncio.run(flow()), parts


def read_csv(csv_bytes):
    return pd.read_csv(BytesIO(csv_bytes), dtype=str)


def emitted_csvs(runtime):
    emitted = runtime.pop_emitted_files()
    return [read_csv(f.file_bytes) for f in emitted]


def test_text_only_produces_chunks():
    elements = [make_text_element("Just plain text.", page_number=1)]
    context = make_context()
    chunks, parts = run_process(
        context, TextPlugin(), partition_result=elements
    )

    assert len(chunks) == 1
    assert chunks[0].plugin == "text"
    assert "Just plain text." in chunks[0].text
    assert chunks[0].metadata["source_page_number"] == 1
    assert parts.runtime.pop_emitted_files() == []


def test_title_strategy_still_used():
    from unstructured.documents.elements import Title

    elements = [
        Title(text="Section One"),
        make_text_element("Body under the section."),
    ]
    context = make_context()
    chunks, parts = run_process(
        context, TextPlugin(), partition_result=elements
    )

    assert len(chunks) == 1
    assert "Section One" in chunks[0].text


def test_consecutive_tables_with_page_break_merge_into_one_csv():
    elements = [
        make_table_element(TABLE_A, page_number=1),
        make_page_break_element(),
        make_table_element(TABLE_A_REPEAT_HEADER, page_number=2),
    ]
    context = make_context(filename="report.pdf")
    chunks, parts = run_process(
        context, TextPlugin(), partition_result=elements
    )

    assert chunks == []

    emitted = parts.runtime.pop_emitted_files()
    assert len(emitted) == 1
    assert emitted[0].filename == "report_table_1.csv"
    assert emitted[0].content_type == "text/csv"

    data = read_csv(emitted[0].file_bytes)
    assert list(data.columns) == ["name", "age"]
    assert data.values.tolist() == [
        ["kyle", "25"],
        ["anne", "21"],
        ["john", "23"],
    ]
    # The repeated header must not survive as a data row.
    assert ["name", "age"] not in [list(row) for row in data.values]


def test_content_between_tables_breaks_the_run():
    elements = [
        make_table_element(TABLE_A, page_number=1),
        make_text_element("Some content here.", page_number=1),
        make_table_element(TABLE_A, page_number=2),
    ]
    context = make_context(filename="report.pdf")

    _, parts = run_process(context, TextPlugin(), partition_result=elements)

    csvs = emitted_csvs(parts.runtime)
    assert len(csvs) == 2
    for csv in csvs:
        assert csv.values.tolist() == [
            ["kyle", "25"],
            ["anne", "21"],
        ]


def test_different_schema_tables_stay_separate():
    elements = [
        make_table_element(TABLE_A, page_number=1),
        make_page_break_element(),
        make_table_element(TABLE_B, page_number=2),
    ]
    context = make_context(filename="report.pdf")

    _, parts = run_process(context, TextPlugin(), partition_result=elements)

    csvs = emitted_csvs(parts.runtime)
    assert len(csvs) == 2
    assert csvs[0].values.tolist() == [
        ["kyle", "25"],
        ["anne", "21"],
    ]
    assert list(csvs[1].columns) == ["part", "length"]


def test_missing_header_continuation_with_same_types_merges():
    elements = [
        make_table_element(TABLE_A, page_number=1),
        make_page_break_element(),
        make_table_element(TABLE_A_NO_HEADER, page_number=2),
    ]
    context = make_context(filename="report.pdf")

    _, parts = run_process(context, TextPlugin(), partition_result=elements)

    csvs = emitted_csvs(parts.runtime)
    assert len(csvs) == 1
    data = csvs[0]
    assert list(data.columns) == ["name", "age"]
    assert data.values.tolist() == [
        ["kyle", "25"],
        ["anne", "21"],
        ["john", "44"],
        ["anika", "16"],
    ]


def test_missing_header_continuation_with_conflicting_types_stays_separate():
    elements = [
        make_table_element(TABLE_A, page_number=1),
        make_page_break_element(),
        make_table_element(TABLE_B_NO_HEADER, page_number=2),
    ]
    context = make_context(filename="report.pdf")

    _, parts = run_process(context, TextPlugin(), partition_result=elements)

    csvs = emitted_csvs(parts.runtime)
    assert len(csvs) == 2


def test_page_furniture_between_tables_does_not_break_the_run():
    elements = [
        make_table_element(TABLE_A, page_number=1),
        make_footer_element("Acme Corp - Confidential", page_number=1),
        make_page_break_element(),
        make_footer_element("Acme Corp - Confidential", page_number=2),
        make_table_element(TABLE_A_REPEAT_HEADER, page_number=2),
    ]
    context = make_context(filename="report.pdf")

    _, parts = run_process(context, TextPlugin(), partition_result=elements)

    csvs = emitted_csvs(parts.runtime)
    assert len(csvs) == 1
    assert csvs[0].values.tolist() == [
        ["kyle", "25"],
        ["anne", "21"],
        ["john", "23"],
    ]


def test_image_is_emitted_as_a_file_and_indexed_by_a_text_chunk():
    image_bytes = b"\xff\xd8\xff\xe0fake-jpeg-bytes"
    elements = [
        make_text_element("An image follows.", page_number=1),
        make_image_element(
            image_bytes,
            page_number=1,
            caption="A chart of sales.",
        ),
        make_text_element("The image is followed by more text.", page_number=1),
    ]
    context = make_context(filename="report.pdf")
    chunks, parts = run_process(
        context, TextPlugin(), partition_result=elements
    )

    image_chunks = [c for c in chunks if "A chart of sales." in c.text]
    assert len(image_chunks) == 1
    image = image_chunks[0]
    assert image.plugin == "text"
    # The chunk carries the description (caption + nearby text), not the bytes.
    assert "A chart of sales." in image.text
    assert "An image follows." in image.text
    assert "followed by more text" in image.text
    assert image.metadata["source_page_number"] == 1
    assert image.metadata["emitted_filename"] == "figure_1.jpg"

    # The bytes ride on an emitted file, not on the chunk.
    emitted = parts.runtime.pop_emitted_files()
    assert len(emitted) == 1
    assert emitted[0].filename == "figure_1.jpg"
    assert emitted[0].content_type == "image/jpeg"
    assert emitted[0].file_bytes == image_bytes
    assert "A chart of sales." in (emitted[0].description or "")


def test_image_without_extracted_bytes_emits_and_indexes_nothing():
    elements = [
        make_text_element("Before."),
        make_image_element(b"ignored"),
        make_text_element("After."),
    ]
    elements[1].metadata.image_base64 = None

    context = make_context(filename="report.docx")
    chunks, parts = run_process(
        context, TextPlugin(), partition_result=elements
    )

    # Only the two text chunks remain; no image chunk, no emitted image file.
    assert all("emitted_filename" not in c.metadata for c in chunks)
    assert parts.runtime.pop_emitted_files() == []


def test_table_only_document_emits_csv_without_description():
    elements = [make_table_element(TABLE_A, page_number=1)]
    context = make_context(filename="report.pdf")
    chunks, parts = run_process(
        context, TextPlugin(), partition_result=elements
    )

    assert chunks == []

    emitted = parts.runtime.pop_emitted_files()
    assert len(emitted) == 1
    # Nothing around the table and no source description: no description.
    assert emitted[0].description is None


def test_emitted_csv_description_carries_nearby_text():
    elements = [
        make_text_element("Sales by region for the first quarter.", page_number=1),
        make_table_element(TABLE_A, page_number=1),
        make_text_element("The figures include all departments.", page_number=1),
    ]
    context = make_context(filename="report.pdf")

    _, parts = run_process(context, TextPlugin(), partition_result=elements)

    emitted = parts.runtime.pop_emitted_files()
    assert len(emitted) == 1
    description = emitted[0].description
    assert "report.pdf" in description
    assert "Sales by region" in description
    assert "all departments" in description


def test_emitted_csv_description_includes_source_description():
    elements = [make_table_element(TABLE_A, page_number=1)]
    context = make_context(
        filename="report.pdf",
        source_description="Q3 sales report for the Nordics region.",
    )

    _, parts = run_process(context, TextPlugin(), partition_result=elements)

    emitted = parts.runtime.pop_emitted_files()
    assert len(emitted) == 1
    assert "Q3 sales report for the Nordics region." in emitted[0].description


def test_description_is_bounded_by_neighbouring_tables():
    elements = [
        make_table_element(TABLE_A, page_number=1),
        make_text_element("Middle note between the tables.", page_number=1),
        make_table_element(TABLE_A, page_number=1),
        make_text_element("Closing note after everything.", page_number=1),
    ]
    context = make_context(filename="report.pdf")

    _, parts = run_process(context, TextPlugin(), partition_result=elements)

    emitted = parts.runtime.pop_emitted_files()
    assert len(emitted) == 2
    first, second = (f.description for f in emitted)

    # The first table sees nothing above and the middle note below.
    assert "Middle note" in first
    assert "Closing note" not in first
    assert "Text before" not in first

    # The second table sees the middle note above and the closing note below.
    assert "Middle note" in second
    assert "Closing note" in second


def test_ignore_images_skips_image_files_and_chunks():
    image_bytes = b"\xff\xd8\xff\xe0fake-jpeg-bytes"
    elements = [
        make_text_element("An image follows.", page_number=1),
        make_image_element(
            image_bytes,
            page_number=1,
            caption="A chart of sales.",
        ),
        make_text_element("The image is followed by more text.", page_number=1),
    ]
    context = make_context(filename="report.pdf")

    chunks, parts = run_process(
        context,
        TextPlugin(ignore_images=True),
        partition_result=elements,
    )

    # Only the text remains (the two blocks chunk together): the figure is
    # neither indexed nor emitted.
    assert len(chunks) == 1
    assert "An image follows." in chunks[0].text
    assert "followed by more text" in chunks[0].text
    assert all("emitted_filename" not in c.metadata for c in chunks)
    assert parts.runtime.pop_emitted_files() == []


def test_ignore_images_still_separates_table_runs():
    # An ignored image produces no output, but it is still content that
    # separates two tables, so the run breaks and two CSVs are emitted.
    elements = [
        make_table_element(TABLE_A, page_number=1),
        make_image_element(b"image bytes", page_number=1),
        make_table_element(TABLE_A, page_number=2),
    ]
    context = make_context(filename="report.pdf")

    _, parts = run_process(
        context,
        TextPlugin(ignore_images=True),
        partition_result=elements,
    )

    csvs = emitted_csvs(parts.runtime)
    assert len(csvs) == 2
    for csv in csvs:
        assert csv.values.tolist() == [
            ["kyle", "25"],
            ["anne", "21"],
        ]


def test_ignore_tables_skips_csv_emission():
    elements = [
        make_text_element("Body text.", page_number=1),
        make_table_element(TABLE_A, page_number=1),
    ]
    context = make_context(filename="report.pdf")

    chunks, parts = run_process(
        context,
        TextPlugin(ignore_tables=True),
        partition_result=elements,
    )

    # The table is not managed: no CSV is emitted, but its text still ends
    # up in the document chunks (make_table_element uses "placeholder"). The
    # basic chunker splits on non-text element categories, so the table text
    # gets its own chunk.
    assert parts.runtime.pop_emitted_files() == []
    texts = " ".join(chunk.text for chunk in chunks)
    assert "Body text." in texts
    assert "placeholder" in texts


def partition_kwargs_for(plugin: TextPlugin, elements) -> dict:
    """Run the plugin on a stubbed partition and return the kwargs it used."""
    context = make_context(filename="report.pdf")
    registry = PluginRegistry()
    registry.register(plugin)
    parts = build_runtime(context, hooks=registry.hooks)

    async def flow():
        with patch.object(text_module, "partition") as mock_partition:
            mock_partition.return_value = elements
            await parts.runtime.hooks.trigger(
                "ingestion_process",
                IngestionProcessPayload(context=context, runtime=parts.runtime),
            )
        return mock_partition.call_args.kwargs

    return asyncio.run(flow())  # type: ignore


def test_ignore_images_disables_image_extraction_in_partition():
    elements = [make_text_element("Body.", page_number=1)]
    kwargs = partition_kwargs_for(TextPlugin(ignore_images=True), elements)

    assert kwargs["extract_images_in_pdf"] is False
    assert kwargs["extract_image_block_types"] is None
    assert kwargs["extract_image_block_to_payload"] is False
    # Table structure inference stays on so ignored table text is readable.
    assert kwargs["infer_table_structure"] is True


def test_default_partition_extracts_images():
    elements = [make_text_element("Body.", page_number=1)]
    kwargs = partition_kwargs_for(TextPlugin(), elements)

    assert kwargs["extract_images_in_pdf"] is True
    assert kwargs["extract_image_block_types"] == ["Image"]
    assert kwargs["extract_image_block_to_payload"] is True
    assert kwargs["infer_table_structure"] is True


def test_defaults_manage_images_and_tables():
    elements = [
        make_table_element(TABLE_A, page_number=1),
        make_image_element(b"image bytes", page_number=1),
    ]
    context = make_context(filename="report.pdf")

    _, parts = run_process(
        context,
        TextPlugin(),
        partition_result=elements,
    )

    emitted = parts.runtime.pop_emitted_files()
    assert [f.filename for f in emitted] == ["figure_1.jpg", "report_table_1.csv"]


def test_rejects_files_it_does_not_accept():
    context = make_context(filename="sales.xlsx")
    registry = PluginRegistry()
    registry.register(TextPlugin())
    parts = build_runtime(context, hooks=registry.hooks)

    def boom(*args, **kwargs):
        raise AssertionError("partition should not be called for rejected files")

    async def flow():
        with patch.object(text_module, "partition", side_effect=boom):
            results = await parts.runtime.hooks.trigger(
                "ingestion_process",
                IngestionProcessPayload(context=context, runtime=parts.runtime),
            )
        return [r for r in results if r]

    assert asyncio.run(flow()) == []
    assert parts.runtime.pop_emitted_files() == []
