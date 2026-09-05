from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.core.config import settings

from app.llm.openai import OpenAIProvider
from app.embedders.sentence_transformers import SentenceTransformerEmbedder

from app.plugin.hooks import HookBus
from app.plugin.registry import PluginRegistry

from app.retrievers.vector import VectorRetriever
from app.retrievers.sparse import SparseRetriever
from app.retrievers.hybrid import HybridRetriever

from app.store_file.local import LocalFileStorage
from app.store_vector.local import LocalVectorStorage
from app.store_sql.local import LocalSqlStorage

from app.services.ingestion import IngestionService
from app.services.retrieval import RetrievalService
from app.services.rag import RagService

from app.plugins.text import TextPlugin
from app.plugins.table import TablePlugin

# Providers
llm = OpenAIProvider(settings)
embedder = SentenceTransformerEmbedder(settings)

# Storage
file_storage = LocalFileStorage(storage_dir=settings.FILE_LOCAL_STORAGE_DIR)

vector_storage = LocalVectorStorage(
    storage_dir=settings.VECTOR_LOCAL_STORAGE_DIR,
    collection_name=settings.VECTOR_COLLECTION_NAME,
)

sql_storage = LocalSqlStorage(storage_dir=settings.SQL_LOCAL_STORAGE_DIR)

# Plugin system
hooks = HookBus()
plugin_registry = PluginRegistry(hooks)
plugin_registry.register(TextPlugin())
plugin_registry.register(TablePlugin())

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

# Services
ingestion_service = IngestionService(
    hooks=hooks,
    registry=plugin_registry,
    llm=llm,
    embedder=embedder,
    vector_storage=vector_storage,
    sql_storage=sql_storage,
    file_storage=file_storage,
)

retrieval_service = RetrievalService(
    retriever=hybrid_retriever,
    llm=llm,
    hooks=hooks,
)

rag_service = RagService(
    llm=llm,
    ingestion_service=ingestion_service,
    retrieval_service=retrieval_service,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await ingestion_service.initialize()

    yield

    await sql_storage.close()
    await vector_storage.close()
