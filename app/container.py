from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.core.config import settings

from app.llm.openai import OpenAIProvider
from app.embedders.sentence_transformers import SentenceTransformerEmbedder

from app.plugins.table import TablePlugin
from app.plugins.text import TextPlugin
from app.retrievers.vector import VectorRetriever
from app.retrievers.sparse import SparseRetriever
from app.retrievers.hybrid import HybridRetriever

from app.store_file.local import LocalFileStorage
from app.store_vector.local import LocalVectorStorage
from app.store_sql.local import LocalSqlStorage

from app.services.rag import RagService

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

rag_service = RagService(
    llm=llm,
    embedder=embedder,
    retriever=hybrid_retriever,
    vector_storage=vector_storage,
    sql_storage=sql_storage,
    file_storage=file_storage,
    plugins=[
        TextPlugin(ignore_images=True),
        TablePlugin(),
    ],
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await rag_service.initialize()

    yield

    await sql_storage.close()
    await vector_storage.close()
