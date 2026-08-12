# llm
from app.llm.gemini import GeminiProvider
from app.embedders.gemini import GeminiEmbedder

llm = GeminiProvider()
embedder = GeminiEmbedder()

# storage
from app.store_file.local import LocalFileStorage
from app.store_metadata.local import LocalMetadataStorage
from app.store_vector.local import LocalVectorStorage

file_storage = LocalFileStorage()
metadata_storage = LocalMetadataStorage()
vector_storage = LocalVectorStorage()

# services
from app.services.document import DocumentService
from app.services.ingestion import IngestionService

document_service = DocumentService(file_storage, metadata_storage)
ingestion_service = IngestionService(document_service, embedder, vector_storage)
