from fastapi import APIRouter
from backend.app.schemas import HealthResponse
from backend.app.core.config import settings

router = APIRouter(tags=["Health"])


@router.get("/health", response_model=HealthResponse)
async def get_health() -> HealthResponse:
    """Return health status, current version, and UTC server timestamp."""
    return HealthResponse(status="ok", version=settings.version)
