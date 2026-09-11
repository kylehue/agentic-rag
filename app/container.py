from contextlib import asynccontextmanager
from functools import partial

from fastapi import FastAPI

from app.agent_tools import build_rag_tools
from app.core.config import settings

from app.llm.openai import OpenAIProvider
from app.embedders.gemini import GeminiEmbedder

from app.plugins.table import TablePlugin
from app.plugins.text import TextPlugin
from app.retrievers.vector import VectorRetriever
from app.retrievers.sparse import SparseRetriever
from app.retrievers.hybrid import HybridRetriever

from app.store_file.local import LocalFileStorage
from app.store_vector.local import LocalVectorStorage
from app.store_sql.local import LocalSqlStorage

from app.services.agent_service import AgentService
from app.services.rag import RagService

# Providers
llm = OpenAIProvider(settings)
embedder = GeminiEmbedder(settings)

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
        TextPlugin(
            ignore_images=True,
            use_api=settings.UNSTRUCTURED_USE_API,
            api_key=settings.UNSTRUCTURED_API_KEY,
        ),
        TablePlugin(),
    ],
)

# The agent service composes the RAG service (its retrieval is the agent's
# search path) with the RAG tools. The tools are passed as a builder: each
# answer gets its own tool set wired to its own evidence-recording
# retrieval path.
agent_service = AgentService(
    rag_service=rag_service,
    llm=llm,
    tools=partial(build_rag_tools, sql_storage=sql_storage, file_storage=file_storage),
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await rag_service.initialize()

    yield

    await sql_storage.close()
    await vector_storage.close()
