from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.core.config import settings

from app.llm.gemini import GeminiProvider
from app.embedders.gemini import GeminiEmbedder

from app.store_file.local import LocalFileStorage
from app.store_vector.local import LocalVectorStorage
from app.store_sql2.local import LocalSqlStorage

from app.services.ingestion import IngestionService
from app.services.retrieval import RetrievalService
from app.services.rag import RagService

from app.retrievers.document import DocumentRetriever
from app.retrievers.image import ImageRetriever
from app.retrievers.spreadsheet import SpreadsheetRetriever

# Providers
llm = GeminiProvider()
embedder = GeminiEmbedder()

# Storage
file_storage = LocalFileStorage(
    settings.FILE_LOCAL_STORAGE_DIR,
)

vector_storage = LocalVectorStorage(
    settings.VECTOR_LOCAL_STORAGE_DIR,
    settings.VECTOR_COLLECTION_NAME,
)

sql_storage = LocalSqlStorage(
    settings.SQL_LOCAL_STORAGE_DIR,
)

# Services
ingestion_service = IngestionService(
    embedder,
    llm,
    file_storage,
    vector_storage,
    sql_storage,
)

retrieval_service = RetrievalService(
    embedder,
    [
        DocumentRetriever(),
        SpreadsheetRetriever(
            llm,
            sql_storage,
        ),
        ImageRetriever(),
    ],
    vector_storage,
    sql_storage,
)

rag_service = RagService(
    ingestion_service,
    retrieval_service,
    llm,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await ingestion_service.initialize()

    yield

    await sql_storage.close()
    await vector_storage.close()
