from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=("../.env", ".env"), extra="ignore")

    database_url: str = "sqlite:///./asigner.db"
    session_secret: str = "dev-only-secret"
    google_client_id: str = ""
    google_client_secret: str = ""
    frontend_url: str = "http://localhost:3000"
    ollama_url: str = "http://localhost:11434"
    sentence_model: str = ""


@lru_cache
def get_settings() -> Settings:
    return Settings()
