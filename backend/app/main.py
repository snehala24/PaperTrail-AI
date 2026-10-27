from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.app.core.config import settings
from backend.app.retrieval.hybrid import ensure_retriever_indexed
from backend.app.api.health import router as health_router
from backend.app.api.documents import router as documents_router
from backend.app.api.search import router as search_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan context manager for startup and shutdown tasks."""
    # Startup: restore in-memory retrieval index from persistent storage if available
    ensure_retriever_indexed()
    yield


app = FastAPI(
    title=settings.app_name,
    version=settings.version,
    description="Multi-Paper Research Evidence Workspace with Claim-to-Evidence Ledger.",
    lifespan=lifespan,
)

# CORS middleware for Next.js frontend communication (from ALLOWED_ORIGINS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API Routers
app.include_router(health_router)
app.include_router(documents_router)
app.include_router(search_router)
