"""
Main FastAPI application for Enterprise Knowledge Assistant
"""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import logging
import sys
sys.path.append('../..')

from shared.config import get_settings
from shared.utils import setup_logging
from .routes import router
from .models import ErrorResponse

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan"""
    # Startup
    logger.info("Starting Enterprise Knowledge Assistant API")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Enterprise Knowledge Assistant API")


# Create FastAPI app
app = FastAPI(
    title="Enterprise Knowledge Assistant API",
    description="API for searching and managing enterprise knowledge",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routes
app.include_router(router, prefix="/api/v1", tags=["Enterprise Knowledge"])


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": "Enterprise Knowledge Assistant API",
        "version": "1.0.0",
        "status": "healthy",
        "docs": "/docs"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": "2024-01-01T00:00:00Z"
    }


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions"""
    logger.error(f"Unhandled exception: {exc}")
    
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error_type=type(exc).__name__,
            error_message=str(exc)
        ).dict()
    )


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8001,
        reload=True
    )