"""
Main FastAPI application
"""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import logging
import sys

from config.settings import settings
from config.api_config import APIConfig
from api.routes import health_router, agent_router, admin_router
from api.middleware import (
    limiter,
    rate_limit_exceeded_handler,
    validation_exception_handler,
    general_exception_handler
)
from agents.agent_factory import AgentFactory
from vector_stores.store_factory import VectorStoreFactory
from monitoring.logging_config import setup_logging
from slowapi.errors import RateLimitExceeded
from fastapi.exceptions import RequestValidationError

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events"""
    # Startup
    logger.info(f"Starting {settings.app_name} v{settings.app_version}")
    logger.info(f"Environment: {settings.environment}")
    
    try:
        # Initialize vector store
        logger.info("Initializing vector store...")
        await VectorStoreFactory.get_vector_store()
        logger.info("Vector store initialized")
        
        # Initialize agents
        logger.info("Initializing agents...")
        await AgentFactory.initialize_all_agents()
        logger.info("Agents initialized")
        
        logger.info("Application startup complete")
    
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        sys.exit(1)
    
    yield
    
    # Shutdown
    logger.info("Shutting down application...")
    
    try:
        # Close vector store
        await VectorStoreFactory.close()
        logger.info("Vector store closed")
    
    except Exception as e:
        logger.error(f"Shutdown error: {e}")
    
    logger.info("Application shutdown complete")


# Create FastAPI app
app = FastAPI(
    title=APIConfig.API_TITLE,
    description=APIConfig.API_DESCRIPTION,
    version=APIConfig.API_VERSION,
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=APIConfig.CORS_ORIGINS,
    allow_credentials=APIConfig.CORS_ALLOW_CREDENTIALS,
    allow_methods=APIConfig.CORS_ALLOW_METHODS,
    allow_headers=APIConfig.CORS_ALLOW_HEADERS,
)

# Add rate limiting
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, rate_limit_exceeded_handler)

# Add error handlers
app.add_exception_handler(RequestValidationError, validation_exception_handler)
app.add_exception_handler(Exception, general_exception_handler)

# Include routers
app.include_router(
    health_router,
    tags=["Health"]
)

app.include_router(
    agent_router,
    prefix=f"{APIConfig.API_PREFIX}/agents",
    tags=["Agents"]
)

app.include_router(
    admin_router,
    prefix=f"{APIConfig.API_PREFIX}/admin",
    tags=["Admin"]
)


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "status": "running",
        "docs_url": "/docs",
        "health_url": "/health"
    }


@app.get("/info")
async def info():
    """Application information"""
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "environment": settings.environment,
        "debug": settings.debug,
        "vector_store": settings.vector_store.value
    }


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.api_reload,
        workers=1 if settings.api_reload else settings.api_workers,
        log_level=settings.log_level.lower()
    )