import os
from pathlib import Path
from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import field_validator

# Points to the 'backend' folder
BASE_DIR = Path(__file__).resolve().parent.parent

class Settings(BaseSettings):
    API_PREFIX: str = "/api"
    DEBUG: bool = False
    
    # SQLite / Postgres Database URL
    DATABASE_URL: str = "sqlite:///./test.db" 
    
    # Allowed CORS Origins
    ALLOWED_ORIGINS: str = "*"
    
    # API Keys (Placeholders allowed until configured by user)
    GEMINI_API_KEY: str = ""
    ELEVENLABS_API_KEY: str = ""

    # Chroma Vector DB Settings for RAG
    CHROMA_PERSIST_DIRECTORY: str = str(BASE_DIR / "chroma_db")
    CHROMA_COLLECTION_NAME: str = "unmute_user_context"

    @field_validator("ALLOWED_ORIGINS")
    def parse_allowed_origins(cls, v: str) -> List[str]:
        if v == "*": return ["*"]
        return v.split(",") if v else []

    model_config = SettingsConfigDict(
        env_file=(str(BASE_DIR / ".env"), ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=True,
    )

settings = Settings()
