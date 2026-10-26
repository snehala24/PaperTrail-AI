import os
from pathlib import Path
from typing import List, Optional
from pydantic import BaseModel

# Load .env from repository root using absolute path
REPO_ROOT = Path(__file__).resolve().parents[3]
ENV_PATH = REPO_ROOT / ".env"


def load_env_file(path: Path) -> None:
    """Read a .env file and set variables not already present in os.environ."""
    if not path.exists():
        return
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, val = line.split("=", 1)
                key = key.strip()
                val = val.strip().strip("'\"")
                if key and key not in os.environ:
                    os.environ[key] = val
    except Exception:
        pass


load_env_file(ENV_PATH)


class Settings(BaseModel):
    """Application configuration loaded from environment variables."""
    app_name: str = "PaperTrail AI"
    version: str = "0.1.0"
    environment: str = os.getenv("ENVIRONMENT", "development")
    gemini_api_key: Optional[str] = os.getenv("GEMINI_API_KEY")
    gemini_model: Optional[str] = os.getenv("GEMINI_MODEL")
    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = int(os.getenv("PORT", "8000"))
    storage_dir: str = os.getenv("STORAGE_DIR", "./data")
    eval_dir: str = os.getenv("EVAL_DIR", "./eval")
    max_upload_mb: int = int(os.getenv("MAX_UPLOAD_MB", "25"))
    max_pages: int = int(os.getenv("MAX_PAGES", "150"))
    allowed_origins: List[str] = [
        origin.strip()
        for origin in os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000").split(",")
        if origin.strip()
    ]


settings = Settings()


def get_gemini_model() -> str:
    """
    Retrieve configured GEMINI_MODEL.
    Fails clearly if GEMINI_MODEL is missing from environment.
    """
    if not settings.gemini_model:
        raise ValueError(
            "GEMINI_MODEL is not set in environment or .env. Please configure GEMINI_MODEL."
        )
    return settings.gemini_model
