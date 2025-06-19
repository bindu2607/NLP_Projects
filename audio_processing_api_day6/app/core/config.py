"""
Advanced configuration management using Pydantic BaseSettings.
Supports multiple environments and secure credential handling.
"""
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List

class Settings(BaseSettings):
    APP_NAME: str = "Advanced Audio Processing API"
    VERSION: str = "2.0.0"
    DEBUG: bool = False
    ENVIRONMENT: str = "development"

    HOST: str = "0.0.0.0"
    PORT: int = 8000

    SECRET_KEY: str = "your-super-secret-key-change-in-production"
    JWT_SECRET: str = "jwt-secret-key-change-in-production"
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7

    REDIS_URL: str = "redis://localhost:6379/0"

    ASR_MODEL_NAME: str = "base"
    ASR_DEVICE: str = "cpu"
    TTS_MODEL_NAME: str = "tts_models/multilingual/multi-dataset/your_tts"
    TRANSLATION_MODEL: str = "Helsinki-NLP/opus-mt-en-fr"

    MAX_UPLOAD_SIZE_MB: int = 50
    ALLOWED_AUDIO_FORMATS: List[str] = ["wav", "mp3", "ogg", "flac", "m4a"]
    PROCESSING_TIMEOUT: int = 300

    LOG_LEVEL: str = "INFO"
    REPORTS_DIR: str = "reports"
    ENABLE_METRICS: bool = True

    ALLOWED_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:8501",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:8501"
    ]

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

@lru_cache()
def get_settings() -> Settings:
    return Settings()
