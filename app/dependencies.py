from app.core.config import settings

# llm
from app.llm.gemini import GeminiProvider
from app.embedders.gemini import GeminiEmbedder

llm = GeminiProvider()
embedder = GeminiEmbedder()

# storage
from app.store_file.local import LocalFileStorage
from app.store_vector.local import LocalVectorStorage
from app.store_sql2.local import LocalSqlStorage

file_storage = LocalFileStorage(settings.FILE_LOCAL_STORAGE_DIR)
vector_storage = LocalVectorStorage(
    settings.VECTOR_LOCAL_STORAGE_DIR,
    settings.VECTOR_COLLECTION_NAME,
)
sql_storage = LocalSqlStorage(settings.SQL_LOCAL_STORAGE_DIR)

# services
from app.services.ingestion import IngestionService
from app.services.retrieval import RetrievalService
from app.services.rag import RagService
from app.retrievers.document import DocumentRetriever
from app.retrievers.image import ImageRetriever
from app.retrievers.spreadsheet import SpreadsheetRetriever

ingestion_service = IngestionService(
    embedder,
    llm,
    vector_storage,
    sql_storage,
)

retrieval_service = RetrievalService(
    embedder,
    [
        DocumentRetriever(),
        SpreadsheetRetriever(llm, sql_storage),
        ImageRetriever(),
    ],
    vector_storage,
    sql_storage,
)
rag_service = RagService(ingestion_service, retrieval_service, llm)
