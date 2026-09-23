from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.agent_tools import RAG_TOOLSET
from app.core.config import settings
from app.embedders.clip import ClipImageEmbedder
from app.embedders.fastembed import FastEmbedder
from app.mcp import McpClient

from app.llm.openai import OpenAIProvider
from app.llm.gemini import GeminiProvider
from app.embedders.gemini import GeminiEmbedder

from app.plugins.image import ImagePlugin
from app.plugins.table import TablePlugin
from app.plugins.text import TextPlugin
from app.rerankers.fastembed import FastReranker
from app.retrievers.text_vector import TextVectorRetriever
from app.retrievers.image_vector import ImageVectorRetriever
from app.retrievers.sparse import SparseRetriever
from app.retrievers.hybrid import HybridRetriever

from app.store_file.local import LocalFileStorage
from app.store_vector.local import LocalVectorStorage
from app.store_sql.local import LocalSqlStorage

from app.services.auth import AuthService
from app.services.chat import ChatService
from app.services.rag import RagService
from app.services.rag_agent import RagAgentService

# LLM Providers
# gemini_llm = GeminiProvider(
#     api_key=settings.GOOGLE_API_KEY,
#     model=settings.GEMINI_MODEL,
# )

openai_llm = OpenAIProvider(
    api_key=settings.OPENAI_API_KEY,
    model=settings.OPENAI_MODEL,
    base_url=settings.OPENAI_BASE_URL,
)

# openrouter_llm = OpenAIProvider(
#     api_key=settings.OPENROUTER_API_KEY,
#     model=settings.OPENROUTER_MODEL,
#     base_url=settings.OPENROUTER_BASE_URL,
# )

# Embedders
# text_embedder = GeminiEmbedder(
#     api_key=settings.GOOGLE_API_KEY,
#     model=settings.GEMINI_EMBEDDING_MODEL,
#     batch_size=settings.EMBEDDING_BATCH_SIZE,
# )
text_embedder = FastEmbedder(
    model_name=settings.FASTEMBED_MODEL,
    batch_size=settings.EMBEDDING_BATCH_SIZE,
    cache_dir=settings.MODEL_CACHE_DIR,
)
# The image embedder is a CLIP cross-encoder: its text and vision encoders
# share a space, so a text query can be matched against image chunks.
image_embedder = ClipImageEmbedder(
    text_model=settings.CLIP_TEXT_MODEL,
    image_model=settings.CLIP_IMAGE_MODEL,
    cache_dir=settings.MODEL_CACHE_DIR,
)

# Storage
file_storage = LocalFileStorage(storage_dir=settings.FILE_LOCAL_STORAGE_DIR)

text_vector_storage = LocalVectorStorage(
    storage_dir=settings.VECTOR_LOCAL_STORAGE_DIR,
    collection_name=settings.VECTOR_COLLECTION_NAME,
)

image_vector_storage = LocalVectorStorage(
    storage_dir=settings.VECTOR_LOCAL_STORAGE_DIR,
    collection_name=settings.IMAGE_VECTOR_COLLECTION_NAME,
)

sql_storage = LocalSqlStorage(storage_dir=settings.SQL_LOCAL_STORAGE_DIR)

# Retrievers
text_vector_retriever = TextVectorRetriever(
    embedder=text_embedder,
    text_vector_storage=text_vector_storage,
    sql_storage=sql_storage,
    top_k=50,
)

image_vector_retriever = ImageVectorRetriever(
    embedder=image_embedder,
    image_vector_storage=image_vector_storage,
    sql_storage=sql_storage,
    top_k=50,
)

sparse_retriever = SparseRetriever(
    sql_storage=sql_storage,
    top_k=50,
)

hybrid_retriever = HybridRetriever(
    retrievers=[
        text_vector_retriever,
        image_vector_retriever,
        sparse_retriever,
    ],
    top_k=50,
)

# Re-ranker
reranker = FastReranker(
    model_name=settings.FASTEMBED_RERANK_MODEL,
    cache_dir=settings.MODEL_CACHE_DIR,
)

# Services
rag_service = RagService(
    llm=openai_llm,
    text_embedder=text_embedder,
    image_embedder=image_embedder,
    retriever=hybrid_retriever,
    reranker=reranker,
    retrieval_top_k=settings.RETRIEVAL_TOP_K,
    text_vector_storage=text_vector_storage,
    image_vector_storage=image_vector_storage,
    sql_storage=sql_storage,
    file_storage=file_storage,
    plugins=[
        TextPlugin(
            ignore_images=False,
            use_api=settings.UNSTRUCTURED_USE_API,
            api_key=settings.UNSTRUCTURED_API_KEY,
            strategy=settings.TEXT_PARTITION_STRATEGY,
        ),
        TablePlugin(llm=openai_llm),
        ImagePlugin(
            use_llm_description=settings.IMAGE_USE_LLM_DESCRIPTION,
            llm=openai_llm,
        ),
    ],
)

auth_service = AuthService(sql_storage=sql_storage)
chat_service = ChatService(sql_storage=sql_storage)

mcp_client = McpClient(settings.MCP_SERVERS) if settings.MCP_SERVERS else None

rag_agent_service = RagAgentService(
    rag_service=rag_service,
    tools=[RAG_TOOLSET],
    mcp_client=mcp_client,
    checkpoint_dir=settings.AGENT_LOCAL_STORAGE_DIR,
    max_tool_rounds=settings.AGENT_MAX_TOOL_ROUNDS,
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
    await text_vector_storage.close()
    await image_vector_storage.close()
