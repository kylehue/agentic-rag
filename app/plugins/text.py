import asyncio
from io import StringIO
from pathlib import Path
from typing import Sequence

import pandas as pd
from unstructured.chunking.basic import chunk_elements
from unstructured.chunking.title import chunk_by_title
from unstructured.documents.elements import Element, Table

from app.models.chunk import IngestedChunk
from app.plugin.base import Plugin
from app.plugin.context import IngestionContext
from app.plugin.runtime import IngestionRuntime


class TextPlugin(Plugin):
    """Handles text documents (pdf, docx, txt, md, ...).

    Chunks the parsed text and emits embedded tables as CSV files so they
    are ingested as a subprocess by the TablePlugin.
    """

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

    @property
    def name(self) -> str:
        return "text"

    @property
    def uses_elements(self) -> bool:
        return True

    def accepts(self, context: IngestionContext) -> bool:
        extension = Path(context.source_filename).suffix.lower().lstrip(".")
        return extension in self.SUPPORTED_EXTENSIONS

    async def process(
        self,
        context: IngestionContext,
        runtime: IngestionRuntime,
    ) -> list[IngestedChunk]:
        text_elements: list[Element] = []
        table_elements: list[Table] = []

        for element in context.elements:
            if isinstance(element, Table):
                table_elements.append(element)
            else:
                text_elements.append(element)

        # TODO:
        # We shouldn't blindly emit files for tables because they could
        # be split into different chunks. If we want better chunking for tables
        # in PDFs, we must collect all tables in one piece.
        # Ignoring for now. If the user expects good results for tables,
        # they should use spreadsheet formats.

        # stem = Path(context.source_filename).stem

        # for index, table in enumerate(table_elements, start=1):
        #     csv_bytes = await asyncio.to_thread(
        #         self._table_to_csv_bytes,
        #         table,
        #     )

        #     if csv_bytes is None:
        #         # Keep the table in the text chunks if it cannot be converted.
        #         text_elements.append(table)
        #         continue

        #     await runtime.emit_file(
        #         filename=f"{stem}_table_{index}.csv",
        #         content_type="text/csv",
        #         file_bytes=csv_bytes,
        #     )

        if not text_elements:
            return []

        raw_chunks = await asyncio.to_thread(
            self._chunk_elements,
            text_elements,
        )

        chunks = [
            IngestedChunk(
                plugin=self.name,
                text=chunk.text,
                metadata={
                    "chunk_source_page_number": getattr(
                        chunk.metadata,
                        "page_number",
                        None,
                    ),
                },
                orig_elements=chunk.metadata.orig_elements or [],
            )
            for chunk in raw_chunks
        ]

        await runtime.save_chunks(chunks)

        return chunks

    @staticmethod
    def _chunk_elements(elements: Sequence[Element]) -> list[Element]:
        has_title = any(type(element).__name__ == "Title" for element in elements)

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

    # @staticmethod
    # def _table_to_csv_bytes(table: Table) -> bytes | None:
    #     """Convert a parsed table element into CSV bytes, or None on failure."""
    #     html = getattr(
    #         table.metadata,
    #         "text_as_html",
    #         None,
    #     )

    #     if not isinstance(html, str) or not html.strip():
    #         return None

    #     try:
    #         tables = pd.read_html(StringIO(html), header=0)
    #     except Exception:
    #         return None

    #     if not tables:
    #         return None

    #     return tables[0].to_csv(index=False).encode("utf-8")
