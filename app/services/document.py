from fastapi import UploadFile

from app.services.ingestion import IngestionService


class DocumentService:
    def __init__(self):
        self.ingestion = IngestionService()

    async def upload(self, file: UploadFile): ...

    async def list(self): ...

    async def get(self, document_id: str): ...

    async def delete(self, document_id: str): ...
