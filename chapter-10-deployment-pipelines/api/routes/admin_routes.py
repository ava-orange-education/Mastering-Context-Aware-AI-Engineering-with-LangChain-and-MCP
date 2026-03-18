"""
Admin and management routes
"""

from fastapi import APIRouter, Depends, HTTPException, status
from typing import Dict
import logging

from api.models.response_models import MetricsResponse
from api.middleware.auth_middleware import require_auth
from api.middleware.rate_limit_middleware import limiter, RateLimitConfig
from vector_stores.store_factory import VectorStoreFactory
from agents.agent_factory import AgentFactory

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/stats", response_model=MetricsResponse)
@limiter.limit(RateLimitConfig.DEFAULT)
async def get_system_stats(
    user: Dict = Depends(require_auth)
):
    """Get system statistics (requires authentication)"""
    
    # In production, these would come from a metrics database
    # This is a placeholder implementation
    
    return MetricsResponse(
        total_queries=1000,
        average_response_time_ms=250.5,
        success_rate=0.98,
        queries_last_hour=45,
        queries_last_24h=890,
        agent_usage={
            "multi_agent": 750,
            "retrieval": 150,
            "analysis": 50,
            "synthesis": 50
        }
    )


@router.get("/vector-store/stats")
@limiter.limit(RateLimitConfig.DEFAULT)
async def get_vector_store_stats(
    user: Dict = Depends(require_auth)
):
    """Get vector store statistics"""
    
    try:
        vector_store = await VectorStoreFactory.get_vector_store()
        stats = await vector_store.get_stats()
        
        return {
            "status": "success",
            "stats": stats
        }
    
    except Exception as e:
        logger.error(f"Failed to get vector store stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get vector store stats: {str(e)}"
        )


@router.post("/vector-store/reindex")
@limiter.limit("1/hour")
async def trigger_reindex(
    user: Dict = Depends(require_auth)
):
    """Trigger vector store reindex (admin only)"""
    
    # Check if user has admin role
    if user.get("payload", {}).get("role") != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    
    # In production, this would trigger an async reindex job
    return {
        "status": "success",
        "message": "Reindex triggered",
        "job_id": "placeholder-job-id"
    }


@router.get("/agents/status")
@limiter.limit(RateLimitConfig.DEFAULT)
async def get_agents_status(
    user: Dict = Depends(require_auth)
):
    """Get status of all agents"""
    
    try:
        health_results = await AgentFactory.health_check_all()
        
        return {
            "status": "success",
            "agents": {
                agent_name: "healthy" if is_healthy else "unhealthy"
                for agent_name, is_healthy in health_results.items()
            }
        }
    
    except Exception as e:
        logger.error(f"Failed to get agent status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get agent status: {str(e)}"
        )


@router.post("/cache/clear")
@limiter.limit("5/hour")
async def clear_cache(
    user: Dict = Depends(require_auth)
):
    """Clear application cache"""
    
    # In production, this would clear Redis or other cache
    return {
        "status": "success",
        "message": "Cache cleared"
    }