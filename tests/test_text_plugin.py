import asyncio

from app.core.config import settings
from app.plugin.hooks import HOOK_FILE_EMITTED, HOOK_CHUNK_SAVED

from fakes import (
    build_runtime,
    make_context,
    make_table_element,
    make_text_element,
)
from app.plugins.text import TextPlugin

TABLE_HTML = (
    "<table>"
    "<tr><td>region</td><td>amount</td></tr>"
    "<tr><td>north</td><td>10</td></tr>"
    "<tr><td>south</td><td>20</td></tr>"
    "</table>"
)


def test_process_chunks_text_and_emits_tables():
    elements = [
        make_text_element("First paragraph of the report.", page_number=1),
        make_table_element(TABLE_HTML, page_number=1),
        make_text_element("Second paragraph with more context.", page_number=2),
    ]
    context = make_context(filename="report.txt", elements=elements)
    parts = build_runtime(context)
    plugin = TextPlugin()

    chunks = asyncio.run(plugin.process(context, parts.runtime))

    # The table is removed from the text chunks and emitted as a CSV file.
    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.plugin == "text"
    assert "First paragraph" in chunk.text
    assert "region" not in chunk.text
    assert chunk.metadata["chunk_source_page_number"] == 1

    emitted = parts.runtime.take_emitted_files()
    assert len(emitted) == 1
    assert emitted[0].filename == "report_table_1.csv"
    assert emitted[0].content_type == "text/csv"
    assert emitted[0].source_id == context.source_id
    assert b"region,amount" in emitted[0].file_bytes
    assert b"north,10" in emitted[0].file_bytes

    # The plugin persisted its own chunk through the runtime.
    assert parts.vector_storage.added
    assert parts.vector_storage.added[0][0] == [chunk.chunk_id]

    chunk_rows = parts.sql_storage.tables[settings.CHUNK_TABLE_NAME]
    assert len(chunk_rows) == 1
    assert chunk_rows[0]["source_id"] == context.source_id
    assert chunk_rows[0]["plugin"] == "text"
    assert chunk_rows[0]["metadata"]["chunk_source_id"] == context.source_id


def test_no_tables_no_emissions():
    elements = [make_text_element("Just plain text.", page_number=1)]
    context = make_context(elements=elements)
    parts = build_runtime(context)

    chunks = asyncio.run(TextPlugin().process(context, parts.runtime))

    assert len(chunks) == 1
    assert parts.runtime.take_emitted_files() == []


def test_table_without_html_stays_in_text_chunks():
    elements = [
        make_text_element("Body text."),
        make_table_element(""),
    ]
    context = make_context(elements=elements)
    parts = build_runtime(context)

    chunks = asyncio.run(TextPlugin().process(context, parts.runtime))

    assert parts.runtime.take_emitted_files() == []
    assert "Body text." in chunks[0].text


def test_no_text_elements_returns_no_chunks():
    context = make_context(elements=[make_table_element(TABLE_HTML)])
    parts = build_runtime(context)

    chunks = asyncio.run(TextPlugin().process(context, parts.runtime))

    assert chunks == []
    emitted = parts.runtime.take_emitted_files()
    assert len(emitted) == 1


def test_emit_fires_hooks():
    elements = [
        make_text_element("Body text."),
        make_table_element(TABLE_HTML),
    ]
    context = make_context(elements=elements)
    parts = build_runtime(context)
    emitted_events: list[dict] = []
    saved_events: list[dict] = []

    async def on_emitted(**payload):
        emitted_events.append(payload)

    async def on_saved(**payload):
        saved_events.append(payload)

    parts.hooks.register(HOOK_FILE_EMITTED, on_emitted)
    parts.hooks.register(HOOK_CHUNK_SAVED, on_saved)

    asyncio.run(TextPlugin().process(context, parts.runtime))

    assert len(emitted_events) == 1
    assert emitted_events[0]["context"] is context
    assert emitted_events[0]["emitted_file"].filename == "report_table_1.csv"
    assert len(saved_events) == 1
    assert saved_events[0]["chunk"].plugin == "text"


def test_chunking_uses_title_strategy_when_titles_present():
    from unstructured.documents.elements import Title

    title = Title(text="Section One")
    elements = [
        title,
        make_text_element("Body under the section."),
    ]
    context = make_context(elements=elements)
    parts = build_runtime(context)

    chunks = asyncio.run(TextPlugin().process(context, parts.runtime))

    assert len(chunks) == 1
    assert "Section One" in chunks[0].text
