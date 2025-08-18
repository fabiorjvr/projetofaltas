from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    database_url: str
    api_title: str = "Football Fouls Analytics API"
    api_version: str = "0.1.0"
    debug: bool = False
    
    class Config:
        env_file = ".env"

settings = Settings()