import asyncio
from unittest.mock import patch

from app.plugin.hooks import IngestionProcessPayload
from app.plugin.registry import PluginRegistry
from app.plugins import text as text_module

from fakes import build_runtime, make_context, make_table_element, make_text_element
from app.plugins.text import TextPlugin

TABLE_HTML = (
    "<table>"
    "<tr><td>region</td><td>amount</td></tr>"
    "<tr><td>north</td><td>10</td></tr>"
    "<tr><td>south</td><td>20</td></tr>"
    "</table>"
)


def run_process(context, runtime, plugin: TextPlugin, partition_result=None):
    """Register the plugin on the runtime's bus and fire the process hook.

    `partition_result` is what the mocked unstructured partition returns.
    """
    PluginRegistry(runtime.hooks).register(plugin)

    async def flow():
        with patch.object(
            text_module,
            "partition",
            return_value=partition_result,
        ):
            results = await runtime.hooks.emit(
                "ingestion_process",
                IngestionProcessPayload(context=context, runtime=runtime),
            )
        chunks = []
        for result in results:
            if result:
                chunks.extend(result)
        return chunks

    return asyncio.run(flow())


def test_process_chunks_text_and_ignores_tables_for_now():
    # Table emission is disabled for now (see the TODO in TextPlugin):
    # embedded tables are neither emitted nor chunked as text.
    elements = [
        make_text_element("First paragraph of the report.", page_number=1),
        make_table_element(TABLE_HTML, page_number=1),
        make_text_element("Second paragraph with more context.", page_number=2),
    ]
    context = make_context(filename="report.txt")
    parts = build_runtime(context)
    plugin = TextPlugin()

    chunks = run_process(context, parts.runtime, plugin, partition_result=elements)

    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.plugin == "text"
    assert "First paragraph" in chunk.text
    assert "Second paragraph" in chunk.text
    assert chunk.metadata["source_page_number"] == 1

    assert parts.runtime.take_emitted_files() == []


def test_no_tables_produces_text_chunks():
    elements = [make_text_element("Just plain text.", page_number=1)]
    context = make_context()
    parts = build_runtime(context)

    chunks = run_process(context, parts.runtime, TextPlugin(), partition_result=elements)

    assert len(chunks) == 1
    assert parts.runtime.take_emitted_files() == []


def test_table_only_document_produces_no_chunks():
    elements = [make_table_element(TABLE_HTML)]
    context = make_context()
    parts = build_runtime(context)

    chunks = run_process(context, parts.runtime, TextPlugin(), partition_result=elements)

    assert chunks == []
    assert parts.runtime.take_emitted_files() == []


def test_rejects_files_it_does_not_accept():
    context = make_context(filename="sales.csv", source_bytes=b"a,b\n1,2")
    parts = build_runtime(context)

    # The plugin rejects before parsing, so partition is never called.
    chunks = run_process(context, parts.runtime, TextPlugin())

    assert chunks == []
    assert parts.runtime.take_emitted_files() == []


def test_chunking_uses_title_strategy_when_titles_present():
    from unstructured.documents.elements import Title

    elements = [
        Title(text="Section One"),
        make_text_element("Body under the section."),
    ]
    context = make_context()
    parts = build_runtime(context)

    chunks = run_process(context, parts.runtime, TextPlugin(), partition_result=elements)

    assert len(chunks) == 1
    assert "Section One" in chunks[0].text
