from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    DATABASE_URL: str = "postgresql://admin:CHANGE_ME@192.168.5.5:5432/postgres"
    BRAIN_DATABASE_URL: str = ""
    REDIS_HOST: str = "redis"
    REDIS_PORT: int = 6379
    REDIS_PASSWORD: str = ""
    REDIS_DB: int = 0
    OPENAI_API_KEY: str = ""
    OPENROUTER_API_KEY: str = ""
    INTERNAL_API_KEY: str = ""
    JWT_SECRET: str = ""
    LANGFUSE_PUBLIC_KEY: str = ""
    LANGFUSE_SECRET_KEY: str = ""
    LANGFUSE_HOST: str = "https://cloud.langfuse.com"
    OPENROUTER_MODEL: str = "nvidia/nemotron-3-ultra-550b-a55b:free"
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    APP_PORT: int = 8001


settings = Settings()
