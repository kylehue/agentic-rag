# core
from app.core.config import Settings

settings = Settings()

# storage
from app.store_file.local import LocalFileStorage
from app.store_metadata.local import LocalMetadataStorage

file_storage = LocalFileStorage()
metadata_storage = LocalMetadataStorage()

# services
from app.services.document import DocumentService

document_service = DocumentService(file_storage, metadata_storage)
