# backend/app/core/config.py
import os
from functools import lru_cache
from typing import Optional, List
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

def _db_url_default() -> str:
    # prioridade: env minúscula, depois MAIÚSCULA, depois default docker-compose
    return (
        os.getenv("database_url")
        or os.getenv("DATABASE_URL")
        or "postgresql+psycopg2://fouls:fouls@db:5432/fouls"
    )

class Settings(BaseSettings):
    # Database
    database_url: str = Field(default_factory=_db_url_default)
    
    # API
    api_title: str = "Football Fouls Analytics API"
    api_version: str = "0.1.0"
    debug: bool = False
    
    # Security
    secret_key: str = Field(
        default_factory=lambda: os.getenv(
            "SECRET_KEY", 
            "your-super-secret-key-change-this-in-production-please-make-it-very-long-and-random"
        )
    )
    algorithm: str = "HS256"
    access_token_expire_minutes: int = Field(
        default_factory=lambda: int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))
    )
    refresh_token_expire_days: int = Field(
        default_factory=lambda: int(os.getenv("REFRESH_TOKEN_EXPIRE_DAYS", "7"))
    )
    
    # Password validation
    password_min_length: int = 8
    password_require_uppercase: bool = True
    password_require_lowercase: bool = True
    password_require_numbers: bool = True
    password_require_special: bool = True
    
    # Rate Limiting
    redis_url: Optional[str] = Field(
        default_factory=lambda: os.getenv("REDIS_URL", "redis://localhost:6379/0")
    )
    rate_limit_per_minute: int = Field(
        default_factory=lambda: int(os.getenv("RATE_LIMIT_PER_MINUTE", "60"))
    )
    rate_limit_burst: int = Field(
        default_factory=lambda: int(os.getenv("RATE_LIMIT_BURST", "10"))
    )
    
    # MLflow Configuration
    mlflow_tracking_uri: str = Field(
        default_factory=lambda: os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    )
    mlflow_experiment_name: str = Field(
        default_factory=lambda: os.getenv("MLFLOW_EXPERIMENT_NAME", "football-fouls-analytics")
    )
    mlflow_s3_endpoint_url: str = Field(
        default_factory=lambda: os.getenv("MLFLOW_S3_ENDPOINT_URL", "http://localhost:9000")
    )
    mlflow_artifact_root: str = Field(
        default_factory=lambda: os.getenv("MLFLOW_ARTIFACT_ROOT", "s3://mlflow-artifacts/")
    )
    
    # MLflow S3 Configuration (for MinIO)
    aws_access_key_id: str = Field(
        default_factory=lambda: os.getenv("AWS_ACCESS_KEY_ID", "minioadmin")
    )
    aws_secret_access_key: str = Field(
        default_factory=lambda: os.getenv("AWS_SECRET_ACCESS_KEY", "minioadmin123")
    )
    
    # CORS
    backend_cors_origins: List[str] = Field(
        default_factory=lambda: [
            "http://localhost:3000",
            "http://localhost:8000",
            "http://127.0.0.1:3000",
            "http://127.0.0.1:8000",
            "https://localhost:3000",
            "https://localhost:8000",
        ]
    )
    
    # Scraping
    fbref_base_url: str = "https://fbref.com"
    user_agent: str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    request_delay: float = 1.0  # Delay between requests in seconds
    
    # Logging
    log_level: str = Field(
        default_factory=lambda: os.getenv("LOG_LEVEL", "INFO")
    )
    
    # Email (for password reset)
    smtp_tls: bool = True
    smtp_port: Optional[int] = Field(
        default_factory=lambda: int(os.getenv("SMTP_PORT", "587"))
    )
    smtp_host: Optional[str] = Field(
        default_factory=lambda: os.getenv("SMTP_HOST")
    )
    smtp_user: Optional[str] = Field(
        default_factory=lambda: os.getenv("SMTP_USER")
    )
    smtp_password: Optional[str] = Field(
        default_factory=lambda: os.getenv("SMTP_PASSWORD")
    )
    emails_from_email: Optional[str] = Field(
        default_factory=lambda: os.getenv("EMAILS_FROM_EMAIL")
    )
    emails_from_name: Optional[str] = Field(
        default_factory=lambda: os.getenv("EMAILS_FROM_NAME", "Football Fouls Analytics API")
    )
    
    # File uploads
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    allowed_file_types: List[str] = Field(
        default_factory=lambda: ["image/jpeg", "image/png", "image/gif"]
    )
    upload_dir: str = Field(
        default_factory=lambda: os.getenv("UPLOAD_DIR", "uploads")
    )
    
    # Environment
    environment: str = Field(
        default_factory=lambda: os.getenv("ENVIRONMENT", "development")
    )

    # em docker, NÃO usar .env local; env > tudo
    model_config = SettingsConfigDict(
        env_file=None,
        env_ignore_empty=True,
        extra="ignore",
        case_sensitive=False,  # aceita DATABASE_URL e database_url
    )

@lru_cache
def get_settings() -> Settings:
    return Settings()

# para import conveniente
settings = get_settings()