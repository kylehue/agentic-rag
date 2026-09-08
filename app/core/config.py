from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    GOOGLE_API_KEY: str = ""
    GEMINI_MODEL: str = "gemini-3.5-flash-lite"
    GEMINI_EMBEDDING_MODEL: str = "gemini-embedding-001"
    SENTENCE_TRANSFORMER_MODEL: str = "google/embeddinggemma-300m"
    HF_TOKEN: str = ""
    OPENAI_API_KEY: str = ""
    OPENAI_MODEL: str = "auto"
    OPENAI_BASE_URL: str = "https://llmrouter.boyemma.com/v1"
    EMBEDDING_BATCH_SIZE: int = 100
    SQL_LOCAL_STORAGE_DIR: str = ".storage/sql/data.sqlite3"
    FILE_LOCAL_STORAGE_DIR: str = "./.storage/file"
    VECTOR_LOCAL_STORAGE_DIR: str = "./.storage/vector"
    VECTOR_COLLECTION_NAME: str = "document_chunks"
    CHUNK_TABLE_NAME: str = "__chunks__"
    DOCUMENT_METADATA_TABLE_NAME: str = "__documents__"
    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
    )


settings = Settings()
