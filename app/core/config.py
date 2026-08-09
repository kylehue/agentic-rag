from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    GOOGLE_API_KEY: str = ""
    FILE_LOCAL_STORAGE_DIR: str = ""
    METADATA_LOCAL_STORAGE_DIR: str = ""
    VECTOR_LOCAL_STORAGE_DIR: str = ""
    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
    )
