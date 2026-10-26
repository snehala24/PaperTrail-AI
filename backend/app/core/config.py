import os

# Essential Windows runtime flags for PyTorch + FAISS OpenMP compatibility and offline caching
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from typing import Optional
from pydantic import BaseModel


class Settings(BaseModel):
    """Application configuration loaded from environment variables."""
    app_name: str = "PaperTrail AI"
    version: str = "0.1.0"
    environment: str = os.getenv("ENVIRONMENT", "development")
    gemini_api_key: Optional[str] = os.getenv("GEMINI_API_KEY")
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = int(os.getenv("PORT", "8000"))
    storage_dir: str = os.getenv("STORAGE_DIR", "./data")
    eval_dir: str = os.getenv("EVAL_DIR", "./eval")


settings = Settings()
