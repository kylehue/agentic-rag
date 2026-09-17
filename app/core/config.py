from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # api keys
    UNSTRUCTURED_API_KEY: str = ""
    GOOGLE_API_KEY: str = ""
    OPENAI_API_KEY: str = ""
    HF_TOKEN: str = ""

    # llm models
    GEMINI_MODEL: str = "gemini-3.5-flash-lite"
    OPENAI_MODEL: str = "auto"

    # embedding models
    GEMINI_EMBEDDING_MODEL: str = "gemini-embedding-2"
    FASTEMBED_MODEL: str = "BAAI/bge-small-en-v1.5"
    SENTENCE_TRANSFORMER_MODEL: str = "google/embeddinggemma-300m"

    # other configs
    UNSTRUCTURED_USE_API: bool = True  # disable = self-host
    OPENAI_BASE_URL: str = "https://llmrouter.boyemma.com/v1"
    EMBEDDING_BATCH_SIZE: int = 100
    SQL_LOCAL_STORAGE_DIR: str = "./.storage/sql"
    FILE_LOCAL_STORAGE_DIR: str = "./.storage/file"
    VECTOR_LOCAL_STORAGE_DIR: str = "./.storage/vector"
    VECTOR_COLLECTION_NAME: str = "document_chunks"
    CHUNK_TABLE_NAME: str = "__chunks__"
    DOCUMENT_METADATA_TABLE_NAME: str = "__documents__"
    USERS_TABLE_NAME: str = "__users__"
    AUTH_TOKENS_TABLE_NAME: str = "__auth_tokens__"
    CHATS_TABLE_NAME: str = "__chats__"
    AGENT_LOCAL_STORAGE_DIR: str = "./.storage/agent"

    # Comma-separated list of allowed browser origins for CORS, or "*" for any.
    CORS_ORIGINS: str = "*"
    # Set when the app is served over HTTPS only; the session cookie then
    # gets the Secure attribute.
    SESSION_COOKIE_SECURE: bool = False

    # How many ingest jobs the background queue runs concurrently (one job
    # per file, so a batch's files ingest in parallel up to this limit).
    INGEST_WORKERS: int = 2

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
    )


settings = Settings()
