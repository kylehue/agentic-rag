from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    GOOGLE_API_KEY: str = ""
    GEMINI_MODEL: str = "gemini-3.5-flash-lite"
    GEMINI_EMBEDDING_MODEL: str = "gemini-embedding-001"
    EMBEDDING_BATCH_SIZE: int = 100
    SQL_LOCAL_STORAGE_DIR: str = ".storage/sqlite/spreadsheets.sqlite3"
    FILE_LOCAL_STORAGE_DIR: str = "./.storage/file"
    METADATA_LOCAL_STORAGE_DIR: str = "./.storage/metadata"
    VECTOR_LOCAL_STORAGE_DIR: str = "./.storage/vector"
    VECTOR_COLLECTION_NAME: str = "document_chunks"
    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
    )


settings = Settings()
