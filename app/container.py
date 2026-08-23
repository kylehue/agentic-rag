from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.core.config import settings

from app.llm.gemini import GeminiProvider
from app.embedders.gemini import GeminiEmbedder

from app.processors.image import ImageProcessor
from app.processors.hybrid import HybridProcessor
from app.processors.table import TableProcessor
from app.processors.text import TextProcessor
from app.retrievers.vector import VectorRetriever
from app.retrievers.sparse import SparseRetriever
from app.retrievers.hybrid import HybridRetriever

from app.store_file.local import LocalFileStorage
from app.store_vector.local import LocalVectorStorage
from app.store_sql.local import LocalSqlStorage

from app.services.ingestion import IngestionService
from app.services.retrieval import RetrievalService
from app.services.rag import RagService

from app.finalizers.hybrid import HybridFinalizer
from app.finalizers.spreadsheet import SpreadsheetFinalizer

# Providers
llm = GeminiProvider()
embedder = GeminiEmbedder()

# Storage
file_storage = LocalFileStorage(storage_dir=settings.FILE_LOCAL_STORAGE_DIR)

vector_storage = LocalVectorStorage(
    storage_dir=settings.VECTOR_LOCAL_STORAGE_DIR,
    collection_name=settings.VECTOR_COLLECTION_NAME,
)

sql_storage = LocalSqlStorage(storage_dir=settings.SQL_LOCAL_STORAGE_DIR)

# Processors
table_processor = TableProcessor(llm=llm)
image_processor = ImageProcessor(llm=llm)
text_processor = TextProcessor(image_processor=image_processor)
hybrid_processor = HybridProcessor(
    processors=[
        table_processor,
        image_processor,
        text_processor,
    ]
)

# Retrievers
vector_retriever = VectorRetriever(
    embedder=embedder,
    vector_storage=vector_storage,
    sql_storage=sql_storage,
    top_k=25,
)

sparse_retriever = SparseRetriever(
    sql_storage=sql_storage,
    top_k=25,
)

hybrid_retriever = HybridRetriever(
    retrievers=[
        vector_retriever,
        sparse_retriever,
    ],
    top_k=5,
)

# Finalizers
spreadsheet_finalizer = SpreadsheetFinalizer(
    llm=llm,
    sql_storage=sql_storage,
)

hybrid_finalizer = HybridFinalizer(
    finalizers=[spreadsheet_finalizer],
)

# Services
ingestion_service = IngestionService(
    llm=llm,
    embedder=embedder,
    processor=hybrid_processor,
    file_storage=file_storage,
    vector_storage=vector_storage,
    sql_storage=sql_storage,
)

retrieval_service = RetrievalService(
    retriever=hybrid_retriever,
    finalizer=hybrid_finalizer,
)

rag_service = RagService(
    llm=llm,
    file_storage=file_storage,
    sql_storage=sql_storage,
    ingestion_service=ingestion_service,
    retrieval_service=retrieval_service,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await ingestion_service.initialize()

    yield

    await sql_storage.close()
    await vector_storage.close()
