from fastapi import UploadFile
from app.processors.pipeline import process
from app.services.document import DocumentService
from app.embedders.base import Embedder
from app.store_vector.base import VectorStorage
from app.store_metadata.base import MetadataStorage
from app.utils.chunks import serialize_chunk
from app.store_sql.base import SqlStorage, StoredSqlTable, SqlTableData
from app.models.document import DocumentCategory, DocumentChunk
from app.llm.base import LLMProvider
from unstructured.partition.auto import partition
from pathlib import Path
import pandas as pd

CHUNK_METADATA_COLLECTION = "chunks"


class IngestionService:
    """Turns an uploaded file into searchable chunks and optional SQL tables.

    Flow: ingest() > process() > embed_documents() > add_documents()
    """

    def __init__(
        self,
        document_service: DocumentService,
        embedder: Embedder,
        vector_storage: VectorStorage,
        metadata_storage: MetadataStorage,
        llm: LLMProvider,
        sql_storage: SqlStorage | None = None,
    ):
        self.document_service = document_service
        self.embedder = embedder
        self.vector_storage = vector_storage
        self.metadata_storage = metadata_storage
        self.llm = llm
        self.sql_storage = sql_storage

    async def ingest(self, file: UploadFile):
        """Save, partition, process, embed, and index one uploaded document."""
        document = await self.document_service.upload(file)

        elements = partition(
            filename=document.path,
            strategy="hi_res",
            infer_table_structure=True,  # Keep tables as structured HTML, not jumbled text
            extract_image_block_types=["Image"],  # Grab images found in the PDF
            extract_image_block_to_payload=True,  # Store images as base64 data you can actually use
        )

        chunks = await process(document, elements, self.llm)
        if document.category is DocumentCategory.SPREADSHEET and self.sql_storage:
            tables = await self.sql_storage.replace_document_tables(
                document.id,
                self._extract_spreadsheet_tables(document.path),
            )
            self._attach_sql_metadata(chunks, tables)
        embeddings = await self.embedder.embed_documents(
            [chunk.text for chunk in chunks]
        )
        for chunk in chunks:
            await self.metadata_storage.upsert(
                CHUNK_METADATA_COLLECTION, chunk.id, serialize_chunk(chunk)
            )
        await self.vector_storage.add([chunk.id for chunk in chunks], embeddings)

        return document

    @staticmethod
    def _attach_sql_metadata(
        chunks: list[DocumentChunk], tables: list[StoredSqlTable]
    ) -> None:
        """Add each spreadsheet chunk's generated SQL table and column names."""
        tables_by_sheet = {table.source_name.casefold(): table for table in tables}
        for chunk in chunks:
            sheet_name = str(chunk.metadata.get("sheet_name", "")).casefold()
            table = tables_by_sheet.get(sheet_name)
            if table is None and len(tables) == 1:
                table = tables[0]
            if table is None:
                continue
            columns = [column.__dict__ for column in table.columns]
            chunk.metadata["sql_table"] = table.name
            chunk.metadata["sql_columns"] = columns
            chunk.text += (
                "\nSQL table: "
                + table.name
                + "\nSQL columns: "
                + ", ".join(
                    f"{column['source_name']} -> {column['name']} ({column['type']})"
                    for column in columns
                )
            )

    @staticmethod
    def _extract_spreadsheet_tables(path: str | Path) -> list[SqlTableData]:
        """Read an XLSX workbook or CSV file into backend-neutral table data."""
        path = Path(path)
        if path.suffix.casefold() == ".csv":
            sheets = {path.stem or "Sheet1": pd.read_csv(path)}
        else:
            sheets = pd.read_excel(path, sheet_name=None)

        return [
            SqlTableData(
                source_name=str(sheet_name),
                columns=[str(column) for column in frame.columns],
                rows=list(frame.itertuples(index=False, name=None)),
            )
            for sheet_name, frame in sheets.items()
            if len(frame.columns)
        ]
