from __future__ import annotations

import asyncio
import base64
import json
import time
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Sequence

import pandas as pd
from bs4 import BeautifulSoup
from unstructured.chunking.basic import chunk_elements
from unstructured.chunking.title import chunk_by_title
from unstructured.documents.elements import (
    Element,
    Footer,
    Header,
    Image,
    PageBreak,
    PageNumber,
    Table,
    Title,
)
from unstructured.partition.auto import partition
from unstructured.staging.base import elements_from_dicts
from unstructured_client import UnstructuredClient
from unstructured_client.models.operations import (
    CreateJobRequest,
    DownloadJobOutputRequest,
)
from unstructured_client.models.shared import BodyCreateJob, InputFiles

from app.models.chunk import IngestedChunk
from app.plugin.base import Plugin
from app.plugin.context import IngestionContext
from app.plugin.runtime import IngestionRuntime

# How much surrounding document text (in characters) is captured above and
# below each embedded table or image.
NEARBY_TEXT_CHARS = 500

# Page furniture: neither breaks a run of tables nor counts as nearby text.
TRANSPARENT_ELEMENT_TYPES = (PageBreak, PageNumber, Header, Footer)

# The Unstructured Platform API endpoint (fixed by Unstructured; see
# docs.unstructured.io). It is job-based, and requires
# unstructured-client>=0.46.2 to resolve correctly.
UNSTRUCTURED_API_URL = "https://platform-api.transform.unstructured.io/api/v1"

# How often to poll a running API job for its status.
API_JOB_POLL_INTERVAL_SECONDS = 10


class TextPlugin(Plugin):
    """Handles text documents (pdf, docx, txt, md, ...)."""

    SUPPORTED_EXTENSIONS = frozenset(
        {
            "pdf",
            "docx",
            "doc",
            "txt",
            "md",
            "pptx",
            "ppt",
            "html",
            "htm",
            "rtf",
            "odt",
            "epub",
            "rst",
        }
    )

    def __init__(
        self,
        *,
        ignore_images: bool = False,
        ignore_tables: bool = False,
        use_api: bool = False,
        api_key: str = "",
        api_url: str = UNSTRUCTURED_API_URL,
    ) -> None:
        """
        - ``ignore_images`` - turn off the plugin's management of embedded images.
          They are not extracted at all (image extraction is disabled in the
          partition call), so no figure files or description chunks are produced.
        - ``ignore_tables`` - turn off the plugin's management of embedded tables.
          No reassembled CSV is emitted, but the table text still ends up in the
          document chunks (``infer_table_structure`` stays on for that).
        - ``use_api`` - partition with the hosted Unstructured Platform API
          instead of running the (hi_res) partition locally. Both backends use
          the same partition settings and return the same Element objects, so
          this is a pure backend switch (for debugging).
        - ``api_key`` - the API key, used only when ``use_api`` is True.
        - ``api_url`` - the Platform API endpoint (defaults to Unstructured's).
        """
        self._ignore_images = ignore_images
        self._ignore_tables = ignore_tables
        self._use_api = use_api
        self._api_key = api_key
        self._api_url = api_url

    @property
    def name(self) -> str:
        return "text"

    def accepts(self, context: IngestionContext) -> bool:
        extension = Path(context.file.filename).suffix.lower().lstrip(".")
        return extension in self.SUPPORTED_EXTENSIONS

    async def on_ingestion_process(
        self,
        context: IngestionContext,
        runtime: IngestionRuntime,
    ) -> list[IngestedChunk]:
        if not self.accepts(context):
            return []

        elements = await asyncio.to_thread(
            self._partition,
            context,
        )

        text_elements, image_positions, runs = _walk_stream(
            elements,
            ignore_images=self._ignore_images,
            ignore_tables=self._ignore_tables,
        )

        chunks: list[IngestedChunk] = []

        if text_elements:
            raw_chunks = await asyncio.to_thread(
                self._chunk_elements,
                text_elements,
            )
            chunks.extend(
                IngestedChunk(
                    plugin=self.name,
                    text=chunk.text,
                    metadata={
                        "source_page_number": getattr(
                            chunk.metadata,
                            "page_number",
                            None,
                        ),
                    },
                )
                for chunk in raw_chunks
            )

        chunks.extend(
            await _emit_image_chunks(
                runtime,
                elements,
                image_positions,
            )
        )

        stem = Path(context.file.filename).stem
        table_index = 0

        for run in runs:
            merged_tables = merge_table_fragments([fragment for _, fragment in run])

            cursor = 0
            for table in merged_tables:
                table_index += 1

                # Map this table back onto the element stream so the
                # description carries the text around *this* table, not
                # the whole run.
                first_index = run[cursor][0]
                cursor += table.fragment_count
                last_index = run[cursor - 1][0]

                text_above = _nearby_text(elements, first_index, -1, NEARBY_TEXT_CHARS)
                text_below = _nearby_text(elements, last_index, +1, NEARBY_TEXT_CHARS)

                csv_bytes = await asyncio.to_thread(
                    _table_to_csv_bytes,
                    table,
                )

                description = _table_description(
                    context.file.filename,
                    context.file.description,
                    text_above,
                    text_below,
                )

                await runtime.emit_file(
                    filename=f"{stem}_table_{table_index}.csv",
                    content_type="text/csv",
                    file_bytes=csv_bytes,
                    description=description,
                )

        return chunks

    def _partition(self, context: IngestionContext) -> list[Element]:
        # ``infer_table_structure`` stays on even when the tables are
        # ignored: it is what keeps the table text in the chunks readable.
        if self._ignore_images:
            extract_images_in_pdf = False
            extract_image_block_types = None
            extract_image_block_to_payload = False
        else:
            extract_images_in_pdf = True
            extract_image_block_types = ["Image"]
            extract_image_block_to_payload = True

        return self._partition_with(
            file_bytes=context.file.file_bytes,
            file_filename=context.file.filename,
            content_type=context.file.content_type,
            extract_images_in_pdf=extract_images_in_pdf,
            extract_image_block_types=extract_image_block_types,
            extract_image_block_to_payload=extract_image_block_to_payload,
        )

    def _partition_with(
        self,
        *,
        file_bytes: bytes,
        file_filename: str,
        content_type: str,
        extract_images_in_pdf: bool,
        extract_image_block_types: list[str] | None,
        extract_image_block_to_payload: bool,
    ) -> list[Element]:
        """Run the partition on the configured backend.

        The partition settings are equivalent for both backends (hi_res, table
        structure inference, and the image block types); only the target
        differs (hosted API vs. local), so switching is a pure backend change
        that returns the same Element objects either way.
        """
        if self._use_api:
            # The Platform API is job-based; the image block types alone
            # control image extraction there (Base64 lands in the element
            # metadata, as locally).
            return _partition_via_api(
                api_key=self._api_key,
                api_url=self._api_url,
                file=file_bytes,
                filename=file_filename,
                content_type=content_type,
                extract_image_block_types=extract_image_block_types,
            )

        return partition(
            file=BytesIO(file_bytes),
            file_filename=file_filename,
            content_type=content_type,
            strategy="hi_res",
            infer_table_structure=True,  # Needed for the table cell grids.
            extract_images_in_pdf=extract_images_in_pdf,
            extract_image_block_types=extract_image_block_types,
            extract_image_block_to_payload=extract_image_block_to_payload,
        )

    @staticmethod
    def _chunk_elements(elements: Sequence[Element]) -> list[Element]:
        has_title = any(isinstance(element, Title) for element in elements)

        if has_title:
            return chunk_by_title(
                elements,
                max_characters=3000,
                new_after_n_chars=2400,
                overlap=200,
                combine_text_under_n_chars=500,
            )

        return chunk_elements(
            elements,
            max_characters=1500,
            overlap=200,
        )


def _partition_via_api(
    *,
    api_key: str,
    api_url: str,
    file: bytes,
    filename: str,
    content_type: str,
    extract_image_block_types: list[str] | None,
) -> list[Element]:
    """Partition one file with the hosted Unstructured Platform API.

    The Platform API is job-based: create a one-file job with a High Res
    Partitioner node, poll it until it completes, and download the element
    JSON. Returns the same Element objects the local partition produces, so
    the rest of the pipeline is unchanged.
    """
    settings: dict = {
        "strategy": "hi_res",
        "infer_table_structure": True,  # Needed for the table cell grids.
    }
    if extract_image_block_types:
        settings["extract_image_block_types"] = extract_image_block_types

    client = UnstructuredClient(api_key_auth=api_key, server_url=api_url)

    create_response = client.jobs.create_job(
        request=CreateJobRequest(
            body_create_job=BodyCreateJob(
                request_data=json.dumps(
                    {
                        "job_nodes": [
                            {
                                "name": "Partitioner",
                                "type": "partition",
                                "subtype": "unstructured_api",
                                "settings": settings,
                            }
                        ]
                    }
                ),
                input_files=[
                    InputFiles(
                        content=file,
                        file_name=filename,
                        content_type=content_type or "application/octet-stream",
                    )
                ],
            )
        )
    )
    job_info = create_response.job_information
    if job_info is None or job_info.id is None:
        raise RuntimeError(
            "Unstructured API job creation returned no job information."
        )
    job_id = job_info.id

    while True:
        job_info = client.jobs.get_job(request={"job_id": job_id}).job_information
        if job_info is None:
            raise RuntimeError(
                "Unstructured API job status returned no job information."
            )
        status = job_info.status
        if status == "COMPLETED":
            break
        if status in ("FAILED", "STOPPED"):
            raise RuntimeError(
                f"Unstructured API job did not complete successfully: {status}"
            )
        time.sleep(API_JOB_POLL_INTERVAL_SECONDS)

    output_files = job_info.output_node_files or []
    if not output_files or output_files[0].file_id is None:
        raise RuntimeError("Unstructured API job returned no output files")

    download_response = client.jobs.download_job_output(
        request=DownloadJobOutputRequest(
            job_id=job_id, file_id=output_files[0].file_id
        )
    )
    # The output is the file's elements as a JSON list.
    if not isinstance(download_response.any, list):
        raise RuntimeError(
            "Unexpected Unstructured API job output: expected a list of "
            f"elements, got {type(download_response.any).__name__}."
        )
    return elements_from_dicts(download_response.any)


# --- element stream ---


def _walk_stream(
    elements: Sequence[Element],
    *,
    ignore_images: bool = False,
    ignore_tables: bool = False,
) -> tuple[
    list[Element], list[tuple[int, Image]], list[list[tuple[int, TableFragment]]]
]:
    """Split the element stream into text, image positions, and runs of tables.

    A run is a maximal sequence of table elements with only page furniture
    (page breaks, page numbers, headers, footers) between them. Any other
    element -- text, an image, an unparseable table -- ends the current run,
    because that content separates the tables in the document. A bare page
    break does not. Images are returned as (stream index, element) pairs so
    the nearby text around each one can be collected.

    Ignored images produce no output and still separate runs from each
    other; ignored tables are not managed, but their text is kept in
    ``text_elements`` so it still ends up in the document chunks.
    """
    text_elements: list[Element] = []
    image_positions: list[tuple[int, Image]] = []
    runs: list[list[tuple[int, TableFragment]]] = []
    current: list[tuple[int, TableFragment]] | None = None

    def flush() -> None:
        nonlocal current
        if current is not None:
            runs.append(current)
            current = None

    for index, element in enumerate(elements):
        if isinstance(element, Table):
            if ignore_tables:
                text_elements.append(element)
                continue
            fragment = _table_element_to_fragment(element)
            if fragment is None:
                flush()
                continue
            if current is None:
                current = [(index, fragment)]
            else:
                current.append((index, fragment))
        elif isinstance(element, Image):
            if not ignore_images:
                image_positions.append((index, element))
            flush()
        elif isinstance(element, TRANSPARENT_ELEMENT_TYPES):
            continue
        else:
            text_elements.append(element)
            flush()

    flush()

    return text_elements, image_positions, runs


def _table_element_to_fragment(element: Table) -> TableFragment | None:
    """Parse a table element's HTML into a fragment, or None if unusable."""
    html = getattr(element.metadata, "text_as_html", None)

    if not isinstance(html, str) or not html.strip():
        return None

    rows, has_header = _html_table_rows(html)
    rows = [row for row in rows if any(cell for cell in row)]

    if not rows:
        return None

    width = max(len(row) for row in rows)
    rows = [row + [""] * (width - len(row)) for row in rows]

    if has_header:
        header, data = rows[0], rows[1:]
    else:
        header, data = None, rows

    return TableFragment(
        header=header,
        data=data,
        page=element.metadata.page_number,
    )


def _html_table_rows(html: str) -> tuple[list[list[str]], bool]:
    """Parse table HTML into (rows, has_header).

    Returns every row in document order as stripped cell strings (a colspan
    cell repeats to fill its span) and whether the first row is a declared
    header row (via <thead> or <th> cells).
    """
    soup = BeautifulSoup(html, "lxml")
    table = soup.find("table")
    if table is None:
        table = soup

    rows: list[list[str]] = []
    for tr in table.find_all("tr"):
        cells: list[str] = []
        for tag in tr.find_all(["td", "th"], recursive=False):
            text = " ".join(tag.get_text(separator=" ", strip=True).split())
            try:
                span = max(1, int(str(tag.get("colspan", "1"))))
            except (TypeError, ValueError):
                span = 1
            cells.extend([text] * span)
        if cells:
            rows.append(cells)

    if not rows:
        return [], False

    first_tr = table.find("tr")
    has_header = (
        (
            first_tr.find_parent("thead") is not None
            or any(
                tag.name == "th"
                for tag in first_tr.find_all(["td", "th"], recursive=False)
            )
        )
        if first_tr
        else False
    )

    return rows, has_header


def _nearby_text(
    elements: Sequence[Element],
    boundary_index: int,
    direction: int,
    max_chars: int,
) -> str:
    """Collect the document text on one side of an element, up to max_chars.

    ``direction`` of -1 walks toward the start of the document (the text
    above) and +1 toward the end (the text below). Page furniture is
    skipped, and the walk stops at the next table or the edge of the
    document. The result is in document order, trimmed to the characters
    closest to the element.
    """
    collected: list[str] = []
    total = 0
    index = boundary_index + direction

    while 0 <= index < len(elements):
        element = elements[index]

        if isinstance(element, Table):
            break

        if not isinstance(element, TRANSPARENT_ELEMENT_TYPES):
            text = " ".join((element.text or "").split())
            if text:
                collected.append(text)
                total += len(text) + 1
                if total >= max_chars:
                    break

        index += direction

    if direction < 0:
        collected.reverse()

    combined = " ".join(collected)

    if len(combined) > max_chars:
        combined = combined[-max_chars:] if direction < 0 else combined[:max_chars]

    return combined


# --- embedded tables: reassembly ---
#
# A table split across a page break is parsed into several Table elements,
# one per page. merge_table_fragments folds the fragments of a run back into
# whole tables. It is specific to this plugin's Unstructured parsing, so it
# lives here rather than in app/utils.


@dataclass
class TableFragment:
    """One table as parsed from the document.

    ``header`` is the table's header row, or ``None`` when the parser found
    no header (e.g. the continuation of a table across a page break that
    did not repeat its header). ``data`` holds the data rows only -- the
    header row is not repeated inside ``data``.
    """

    header: list[str] | None
    data: list[list[str]]
    page: int | None = None


@dataclass
class MergedTable:
    """The result of merging a run of table fragments.

    ``fragment_count`` is how many source fragments this table was built
    from. Because merging consumes the fragments in order, the counts of
    the returned tables (in order) let a caller map each table back onto
    the fragment sequence it came from.
    """

    header: list[str] | None
    data: list[list[str]]
    page: int | None = None
    fragment_count: int = 1


def _fragment_width(header: list[str] | None, data: list[list[str]]) -> int:
    if header is not None:
        return len(header)
    return max((len(row) for row in data), default=0)


def _normalize_row(row: Sequence[str]) -> tuple[str, ...]:
    return tuple(cell.strip().casefold() for cell in row)


def _is_number(value: str) -> bool:
    """Whether a cell value is a plain number (thousands separators, a
    leading currency symbol, and a trailing percent are tolerated)."""
    v = value.strip().replace(",", "").replace(" ", "")
    v = v.lstrip("$€£¥")
    if v.endswith("%"):
        v = v[:-1]
    if not v:
        return False
    try:
        float(v)
        return True
    except ValueError:
        return False


def _column_types(rows: Sequence[Sequence[str]], index: int) -> set[str]:
    """The set of value kinds ("number" / "text") seen in one column."""
    kinds: set[str] = set()
    for row in rows:
        if index >= len(row):
            continue
        value = row[index].strip()
        if not value:
            continue
        kinds.add("number" if _is_number(value) else "text")
    return kinds


def _value_types_compatible(
    rows_a: Sequence[Sequence[str]],
    rows_b: Sequence[Sequence[str]],
    width: int,
) -> bool:
    """True when no column mixes numbers on one side with non-numeric text
    on the other. Empty cells carry no information and never conflict."""
    for index in range(width):
        types_a = _column_types(rows_a, index)
        types_b = _column_types(rows_b, index)
        if "number" in types_a and "text" in types_b:
            return False
        if "text" in types_a and "number" in types_b:
            return False
    return True


def _is_repeated_header(current: MergedTable, fragment: TableFragment) -> bool:
    """Whether the fragment's header is the table's header repeated."""
    return (
        fragment.header is not None
        and current.header is not None
        and _normalize_row(current.header) == _normalize_row(fragment.header)
    )


def _candidate_rows(current: MergedTable, fragment: TableFragment) -> list[list[str]]:
    """The fragment's rows treated as data, relative to the table.

    A fragment's declared header counts as data unless it repeats the
    table's header. A parser frequently misclassifies a continuation's first
    data row as a header when a table runs across a page break without
    repeating its header, so a non-repeating declared header is always a
    candidate data row -- the value-type check in `_can_merge` then decides
    whether it really belongs to the table.
    """
    rows = [list(row) for row in fragment.data]
    if fragment.header is not None and not _is_repeated_header(current, fragment):
        rows.insert(0, list(fragment.header))
    return rows


def _can_merge(current: MergedTable, fragment: TableFragment) -> bool:
    """Whether a fragment continues the table accumulated so far.

    A fragment continues the table when it has the same number of columns
    and, additionally:

    - it repeats the table's header (a continuation page), or
    - its rows, treated as data, are data-compatible with the table's
      existing rows (no column mixes numbers with non-numeric text). A
      declared header that does not repeat the table's header counts as a
      candidate data row (see `_candidate_rows`).
    """
    width = _fragment_width(current.header, current.data)
    if _fragment_width(fragment.header, fragment.data) != width:
        return False

    if _is_repeated_header(current, fragment):
        return True

    return _value_types_compatible(
        current.data, _candidate_rows(current, fragment), width
    )


def _absorb(current: MergedTable, fragment: TableFragment) -> None:
    """Append the fragment's data rows to the table.

    A repeated header is deliberately not copied: it is the same header the
    table already carries, so only the data rows are added. A declared
    header that does not repeat the table's header is a misclassified
    continuation row, so it is appended as data.
    """
    current.data.extend(_candidate_rows(current, fragment))


def merge_table_fragments(
    fragments: Sequence[TableFragment],
) -> list[MergedTable]:
    """Merge a run of consecutive table fragments into whole tables.

    A "run" is a sequence of table fragments with no other document content
    (text, images, ...) between them -- in particular, a page break does
    not break a run. This function reassembles the fragments of such a run:

    - a fragment whose header repeats the table's header is a continuation
      page; its header is dropped (kept only once) and its rows are
      appended;
    - a fragment without a header is appended as raw data rows when its
      column count matches and its values are type-compatible with the
      table's existing rows;
    - a fragment whose declared header does *not* repeat the table's header
      is treated as a misclassified continuation: its "header" row is a
      candidate data row, and the fragment is appended (header row included)
      when all of its rows are type-compatible with the table's existing
      rows. This recovers page-2 continuations where the parser mistook the
      first data row for a header.
    - anything else starts a new table.

    The fragments are expected in document order.
    """
    tables: list[MergedTable] = []
    current: MergedTable | None = None

    for fragment in fragments:
        if current is not None and _can_merge(current, fragment):
            _absorb(current, fragment)
            current.fragment_count += 1
        else:
            if current is not None:
                tables.append(current)
            current = MergedTable(
                header=list(fragment.header) if fragment.header is not None else None,
                data=[list(row) for row in fragment.data],
                page=fragment.page,
            )

    if current is not None:
        tables.append(current)

    return tables


def _table_to_csv_bytes(table: MergedTable) -> bytes:
    """Serialize a reassembled table to CSV.

    The table always gets a header row (generated ``col_N`` names when none
    was detected) so downstream readers treat row 0 as column names.
    """
    if table.header is not None:
        header = list(table.header)
    else:
        width = max((len(row) for row in table.data), default=0)
        header = [f"col_{index + 1}" for index in range(width)]

    return pd.DataFrame(table.data, columns=header).to_csv(index=False).encode("utf-8")


def _table_description(
    source_filename: str,
    file_description: str | None,
    text_above: str,
    text_below: str,
) -> str | None:
    """Build the description emitted alongside a table, or None if there is
    nothing to say.

    The description rides on the emitted file and reaches the plugins that
    handle the CSV (e.g. the TablePlugin's LLM), which see both the source
    document's description and the text found around the table.
    """
    parts = [f"This table was extracted from the document '{source_filename}'."]

    if file_description:
        parts.append(f"Description of the source document: {file_description}")
    if text_above:
        parts.append(f"Text before the table: {text_above}")
    if text_below:
        parts.append(f"Text after the table: {text_below}")

    if len(parts) == 1:
        return None

    return "\n\n".join(parts)


# --- embedded images ---


async def _emit_image_chunks(
    runtime: IngestionRuntime,
    elements: Sequence[Element],
    image_positions: Sequence[tuple[int, Element]],
) -> list[IngestedChunk]:
    """Emit each extracted image as a file and index a chunk for its text.

    The image bytes are handed to ``runtime.emit_file`` so the file is
    available for persistence/subprocessing instead of riding on the chunk.
    The returned chunk carries the description (caption + nearby text), which
    is what makes the figure searchable; its metadata links back to the
    emitted filename.
    """
    chunks: list[IngestedChunk] = []
    index = 0

    for position, element in image_positions:
        encoded = getattr(element.metadata, "image_base64", None)
        if not isinstance(encoded, str) or not encoded:
            continue

        index += 1
        raw_bytes = base64.b64decode(encoded)
        mime_type = getattr(element.metadata, "image_mime_type", None) or "image/jpeg"
        page_number = element.metadata.page_number
        filename = f"figure_{index}{_extension_for_mime(mime_type)}"

        description = _image_description(
            index,
            element,
            _nearby_text(elements, position, -1, NEARBY_TEXT_CHARS),
            _nearby_text(elements, position, +1, NEARBY_TEXT_CHARS),
        )

        await runtime.emit_file(
            filename=filename,
            content_type=mime_type,
            file_bytes=raw_bytes,
            description=description,
        )

        chunks.append(
            IngestedChunk(
                plugin="text",
                text=description,
                metadata={
                    "source_page_number": page_number,
                    "emitted_filename": filename,
                },
            )
        )

    return chunks


def _image_description(
    index: int,
    element: Element,
    text_above: str,
    text_below: str,
) -> str:
    parts = [f"Figure {index}."]

    caption = " ".join((element.text or "").split())
    if caption:
        parts.append(f"Caption: {caption}")

    if text_above:
        parts.append(f"Nearby text before: {text_above}")
    if text_below:
        parts.append(f"Nearby text after: {text_below}")

    page_number = element.metadata.page_number
    if page_number is not None:
        parts.append(f"Page: {page_number}")

    return "\n".join(parts)


def _extension_for_mime(mime_type: str) -> str:
    return {
        "image/jpeg": ".jpg",
        "image/png": ".png",
        "image/gif": ".gif",
        "image/webp": ".webp",
        "image/bmp": ".bmp",
    }.get(mime_type, ".bin")
