import os
from pathlib import Path
from typing import Any, List, Optional
from pydantic import BaseModel, field_validator

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
    gemini_max_output_tokens: int = int(os.getenv("GEMINI_MAX_OUTPUT_TOKENS", "2048"))
    gemini_thinking_level: Optional[str] = os.getenv("GEMINI_THINKING_LEVEL", "minimal")
    gemini_timeout: float = float(os.getenv("GEMINI_TIMEOUT", "45.0"))
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

    @field_validator("gemini_max_output_tokens", mode="before")
    @classmethod
    def validate_max_output_tokens(cls, v: Any) -> int:
        try:
            val = int(v)
        except (ValueError, TypeError):
            raise ValueError(f"GEMINI_MAX_OUTPUT_TOKENS must be a valid integer, got {v!r}")
        if val <= 0:
            raise ValueError(f"GEMINI_MAX_OUTPUT_TOKENS must be positive, got {val}")
        return val

    @field_validator("gemini_thinking_level", mode="before")
    @classmethod
    def validate_thinking_level(cls, v: Any) -> Optional[str]:
        if v is None:
            return None
        cleaned = str(v).strip().lower()
        if cleaned in ("", "none", "null"):
            return None
        valid_levels = {"minimal", "low", "medium", "high"}
        if cleaned not in valid_levels:
            raise ValueError(
                f"GEMINI_THINKING_LEVEL must be one of {sorted(valid_levels)} or None, got {v!r}"
            )
        return cleaned

    @field_validator("gemini_timeout", mode="before")
    @classmethod
    def validate_timeout(cls, v: Any) -> float:
        try:
            val = float(v)
        except (ValueError, TypeError):
            raise ValueError(f"GEMINI_TIMEOUT must be a valid float, got {v!r}")
        if val <= 0:
            raise ValueError(f"GEMINI_TIMEOUT must be positive, got {val}")
        return val


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


def get_gemini_api_key() -> str:
    """
    Retrieve configured GEMINI_API_KEY.
    Fails clearly if GEMINI_API_KEY is missing from environment.
    """
    if not settings.gemini_api_key:
        raise ValueError(
            "GEMINI_API_KEY is not set in environment or .env. Please configure GEMINI_API_KEY."
        )
    return settings.gemini_api_key


def get_gemini_max_output_tokens() -> int:
    """Retrieve configured GEMINI_MAX_OUTPUT_TOKENS, defaulting to 2048."""
    return settings.gemini_max_output_tokens


def get_gemini_thinking_level() -> Optional[str]:
    """Retrieve configured GEMINI_THINKING_LEVEL, defaulting to 'minimal'."""
    return settings.gemini_thinking_level
