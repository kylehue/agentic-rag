from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.agent_tools import RAG_TOOLSET
from app.core.config import settings
from app.embedders.fastembed import FastEmbedder
from app.mcp import McpClient

from app.llm.gemini import GeminiProvider
from app.embedders.gemini import GeminiEmbedder

from app.plugins.table import TablePlugin
from app.plugins.text import TextPlugin
from app.rerankers.fastembed import FastReranker
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
llm = GeminiProvider(
    api_key=settings.GOOGLE_API_KEY,
    model=settings.GEMINI_MODEL,
)
# embedder = GeminiEmbedder(
#     api_key=settings.GOOGLE_API_KEY,
#     model=settings.GEMINI_EMBEDDING_MODEL,
#     batch_size=settings.EMBEDDING_BATCH_SIZE,
# )
embedder = FastEmbedder(
    model_name=settings.FASTEMBED_MODEL,
    batch_size=settings.EMBEDDING_BATCH_SIZE,
)

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
    top_k=50,
)

sparse_retriever = SparseRetriever(
    sql_storage=sql_storage,
    top_k=50,
)

hybrid_retriever = HybridRetriever(
    retrievers=[
        vector_retriever,
        sparse_retriever,
    ],
    top_k=50,
)

# Re-ranker
reranker = FastReranker(model_name=settings.FASTEMBED_RERANK_MODEL)

rag_service = RagService(
    llm=llm,
    embedder=embedder,
    retriever=hybrid_retriever,
    reranker=reranker,
    retrieval_top_k=settings.RETRIEVAL_TOP_K,
    vector_storage=vector_storage,
    sql_storage=sql_storage,
    file_storage=file_storage,
    plugins=[
        TextPlugin(
            ignore_images=True,
            use_api=settings.UNSTRUCTURED_USE_API,
            api_key=settings.UNSTRUCTURED_API_KEY,
            strategy=settings.TEXT_PARTITION_STRATEGY,
        ),
        TablePlugin(),
    ],
)

# Auth and chats: decoupled from the RAG services; they only need SQL
# storage.
auth_service = AuthService(sql_storage=sql_storage)
chat_service = ChatService(sql_storage=sql_storage)

mcp_client = McpClient(settings.MCP_SERVERS) if settings.MCP_SERVERS else None

rag_agent_service = RagAgentService(
    rag_service=rag_service,
    tools=[RAG_TOOLSET],
    mcp_client=mcp_client,
    checkpoint_dir=settings.AGENT_LOCAL_STORAGE_DIR,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Create the schema (every table, owned by app.database), connect any MCP
    # servers, open the agent's checkpoint database, and start the background
    # ingest workers.
    await sql_storage.create_tables()
    if mcp_client is not None:
        await mcp_client.connect()
    await rag_agent_service.initialize()
    await rag_service.start_ingest_queue()

    yield

    # Stop the ingest workers, then close every resource: the agent's
    # checkpoint connection, the MCP connections, and the SQL and vector
    # stores.
    await rag_service.stop_ingest_queue()
    await rag_agent_service.close()
    if mcp_client is not None:
        await mcp_client.close()
    await sql_storage.close()
    await vector_storage.close()
