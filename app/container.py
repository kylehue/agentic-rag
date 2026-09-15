from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.agent_tools import (
    InspectTableRelationshipsTool,
    InspectTableTool,
    SearchDocumentTool,
    SqlQueryDocumentsTool,
    SqlQueryTableTool,
)
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

from app.services.auth import AuthService
from app.services.chat import ChatService
from app.services.rag import RagService
from app.services.rag_agent import RagAgentService

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

# Auth and chats: decoupled from the RAG services; they only need SQL
# storage.
auth_service = AuthService(sql_storage=sql_storage)
chat_service = ChatService(sql_storage=sql_storage)

rag_agent_service = RagAgentService(
    rag_service=rag_service,
    tools=[
        SearchDocumentTool(),
        InspectTableTool(),
        InspectTableRelationshipsTool(),
        SqlQueryTableTool(),
        SqlQueryDocumentsTool(),
    ],
    checkpoint_dir=settings.AGENT_LOCAL_STORAGE_DIR,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize top-down: the agent service initializes the RAG service it
    # wraps (the system tables); auth and chat manage their own tables.
    await rag_agent_service.initialize()
    await auth_service.initialize()
    await chat_service.initialize()

    yield

    # The composition root closes every database: the checkpoint connection
    # (through the agent service, whose only own resource it is) and the SQL
    # and vector stores.
    await rag_agent_service.close()
    await sql_storage.close()
    await vector_storage.close()
