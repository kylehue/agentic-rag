from fastapi import UploadFile
from app.processors.pipeline import process
from app.services.document import DocumentService
from app.embedders.base import Embedder
from app.store_vector.base import VectorStorage
from unstructured.partition.auto import partition


class IngestionService:
    def __init__(
        self,
        document_service: DocumentService,
        embedder: Embedder,
        vector_storage: VectorStorage,
    ):
        self.document_service = document_service
        self.embedder = embedder
        self.vector_storage = vector_storage

    async def ingest(self, file: UploadFile):
        document = await self.document_service.upload(file)

        elements = partition(
            filename=document.path,
            strategy="hi_res",
            infer_table_structure=True,  # Keep tables as structured HTML, not jumbled text
            extract_image_block_types=["Image"],  # Grab images found in the PDF
            extract_image_block_to_payload=True,  # Store images as base64 data you can actually use
        )

        chunks = await process(document, elements)
        embeddings = await self.embedder.embed_documents(
            [chunk.text for chunk in chunks]
        )
        await self.vector_storage.add_documents(chunks, embeddings)

        return document
