from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    GOOGLE_API_KEY: str = ""
    GEMINI_MODEL: str = "gemini-3.5-flash-lite"
    GEMINI_EMBEDDING_MODEL: str = "gemini-embedding-001"
    EMBEDDING_BATCH_SIZE: int = 100
    FILE_LOCAL_STORAGE_DIR: str = ""
    METADATA_LOCAL_STORAGE_DIR: str = ""
    VECTOR_LOCAL_STORAGE_DIR: str = ""
    VECTOR_COLLECTION_NAME: str = "document_chunks"
    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
    )


settings = Settings()
