"""
Main FastAPI Application

DevOps Monitoring Agent API
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging
import sys
sys.path.append('..')

from .routes import router
from shared.config import get_settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Get settings
settings = get_settings()

# Create FastAPI app
app = FastAPI(
    title="DevOps Monitoring Agent API",
    description="AI-powered DevOps monitoring and incident response",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routes
app.include_router(router, prefix="/api/v1", tags=["monitoring"])


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    
    logger.info("Starting DevOps Monitoring Agent API")
    
    # In production, initialize:
    # - Agents (incident detection, remediation, RCA, etc.)
    # - Integrations (Prometheus, Kubernetes, PagerDuty, etc.)
    # - RAG systems
    # - Monitoring components
    
    logger.info("API initialized successfully")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    
    logger.info("Shutting down DevOps Monitoring Agent API")
    
    # Cleanup resources


@app.get("/")
async def root():
    """Root endpoint"""
    
    return {
        "name": "DevOps Monitoring Agent API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs"
    }


@app.get("/api/v1")
async def api_info():
    """API information"""
    
    return {
        "version": "v1",
        "endpoints": {
            "incidents": "/api/v1/incidents",
            "metrics": "/api/v1/metrics",
            "anomaly_detection": "/api/v1/anomaly-detection",
            "remediation": "/api/v1/remediation",
            "actions": "/api/v1/actions/execute",
            "rca": "/api/v1/rca",
            "monitoring": "/api/v1/monitoring/status",
            "alerts": "/api/v1/alerts",
            "health": "/api/v1/health"
        }
    }


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8003,
        log_level="info"
    )