from typing import Literal

from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.mcp.server import McpServer
from app.plugins.text import PartitionStrategy


class Settings(BaseSettings):
    # api keys
    UNSTRUCTURED_API_KEY: str = ""
    GOOGLE_API_KEY: str = ""
    OPENAI_API_KEY: str = ""
    OPENROUTER_API_KEY: str = ""
    HF_TOKEN: str = ""

    # llm models
    GEMINI_MODEL: str = "gemini-3.5-flash-lite"
    OPENAI_BASE_URL: str = "https://llmrouter.boyemma.com/v1"
    OPENAI_MODEL: str = "auto"
    # The auxiliary OpenRouter LLM (openrouter_llm in the container). Defaults
    # to OpenRouter's free-model router. Used for cheap work like image
    # description, kept separate from the pipeline LLM.
    OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"
    OPENROUTER_MODEL: str = "openrouter/free"

    # embedding models
    GEMINI_EMBEDDING_MODEL: str = "gemini-embedding-2"
    FASTEMBED_MODEL: str = "BAAI/bge-small-en-v1.5"

    # reranker model (fastembed cross-encoder; re-ranks retrieved chunks)
    FASTEMBED_RERANK_MODEL: str = "Xenova/ms-marco-MiniLM-L-6-v2"

    # image embedding models (CLIP, a shared text+image space)
    CLIP_TEXT_MODEL: str = "Qdrant/clip-ViT-B-32-text"
    CLIP_IMAGE_MODEL: str = "Qdrant/clip-ViT-B-32-vision"
    # Have the image plugin ask the LLM to describe each image, using any
    # user-provided description as additional context. When False, only the
    # user-provided description is used (no text chunk if absent).
    IMAGE_USE_LLM_DESCRIPTION: bool = False

    # other configs
    UNSTRUCTURED_USE_API: bool = False  # disable = self-host
    # The text plugin's partitioning strategy, applied to both the local and
    # the hosted API backends.
    TEXT_PARTITION_STRATEGY: PartitionStrategy = "fast"
    EMBEDDING_BATCH_SIZE: int = 100
    AGENT_MAX_TOOL_ROUNDS: int = 24
    # Final number of chunks a retrieval returns. The retrievers fetch a wider
    # candidate pool and the reranker re-ranks it; this caps the result.
    RETRIEVAL_TOP_K: int = 5
    # Where self-hosted (fastembed) models are downloaded and cached.
    MODEL_CACHE_DIR: str = "./.models"
    SQL_LOCAL_STORAGE_DIR: str = "./.storage/sql"
    FILE_LOCAL_STORAGE_DIR: str = "./.storage/file"
    VECTOR_LOCAL_STORAGE_DIR: str = "./.storage/vector"
    VECTOR_COLLECTION_NAME: str = "document_chunks"
    IMAGE_VECTOR_COLLECTION_NAME: str = "image_chunks"
    AGENT_LOCAL_STORAGE_DIR: str = "./.storage/agent"

    # Comma-separated list of allowed browser origins for CORS, or "*" for any.
    CORS_ORIGINS: str = "*"
    # Set when the app is served over HTTPS only; the session cookie then
    # gets the Secure attribute.
    SESSION_COOKIE_SECURE: bool = False

    # How many ingest jobs the background queue runs concurrently (one job
    # per file, so a batch's files ingest in parallel up to this limit).
    INGEST_WORKERS: int = 2

    # MCP servers the agent connects to (client direction); pydantic-settings
    # JSON-decodes the value from the env. Empty means no MCP.
    MCP_SERVERS: list[McpServer] = [
        McpServer(
            name="parallel",
            url="https://search.parallel.ai/mcp",
        )
    ]

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
    )


settings = Settings()
