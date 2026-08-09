from app.store_file.base import FileStorage
from app.store_metadata.base import MetadataStorage
from app.store_vector.base import VectorStorage
from unstructured.partition.auto import partition


class IngestionService:
    def __init__(
        self,
        metadata_storage: MetadataStorage,
        file_storage: FileStorage,
        vector_storage: VectorStorage,
        chunker: Chunker,
        embedding_service: Embedder,
    ):
        self.metadata_storage = metadata_storage
        self.file_storage = file_storage
        self.vector_storage = vector_storage
        self.parser = parser
        self.chunker = chunker
        self.embedding_service = embedding_service

    async def ingest(self, document_id: str):
        document = await self.metadata_storage.get(document_id)

        if document is None:
            raise DocumentNotFoundError(document_id)

        elements = partition(
            filename=document.path,
            strategy="hi_res",
            infer_table_structure=True,  # Keep tables as structured HTML, not jumbled text
            extract_image_block_types=["Image"],  # Grab images found in the PDF
            extract_image_block_to_payload=True,  # Store images as base64 data you can actually use
        )

        await self.metadata_storage.update_status(
            document_id,
            DocumentStatus.PROCESSING,
        )

        try:
            elements = await self.parser.load(document.path)

            chunks = self.chunker.chunk(elements)

            embeddings = await self.embedding_service.embed(
                [chunk.text for chunk in chunks]
            )

            await self.vector_storage.add(
                document=document,
                chunks=chunks,
                embeddings=embeddings,
            )

            await self.metadata_storage.update_status(
                document_id,
                DocumentStatus.COMPLETED,
            )

        except Exception:

            await self.metadata_storage.update_status(
                document_id,
                DocumentStatus.FAILED,
            )

            raise
