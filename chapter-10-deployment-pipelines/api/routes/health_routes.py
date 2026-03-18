"""
Health check and monitoring routes
"""

from fastapi import APIRouter, Depends
import time
import psutil
from datetime import datetime

from api.models.response_models import (
    HealthCheckResponse,
    HealthStatus,
    ComponentHealth
)
from api.models.request_models import HealthCheckRequest
from agents.agent_factory import AgentFactory
from vector_stores.store_factory import VectorStoreFactory
from config.settings import settings

router = APIRouter()

# Track application start time
START_TIME = time.time()


@router.get("/health", response_model=HealthCheckResponse)
async def basic_health_check():
    """Basic health check endpoint"""
    uptime = time.time() - START_TIME
    
    return HealthCheckResponse(
        status=HealthStatus.HEALTHY,
        version=settings.app_version,
        uptime_seconds=uptime,
        components={"api": ComponentHealth(status=HealthStatus.HEALTHY)}
    )


@router.post("/health/detailed", response_model=HealthCheckResponse)
async def detailed_health_check(request: HealthCheckRequest):
    """Detailed health check with component verification"""
    uptime = time.time() - START_TIME
    components = {}
    
    overall_status = HealthStatus.HEALTHY
    
    # Check API
    components["api"] = ComponentHealth(
        status=HealthStatus.HEALTHY,
        message="API is running"
    )
    
    # Check vector store if requested
    if request.check_vector_store:
        try:
            start = time.time()
            vector_store = await VectorStoreFactory.get_vector_store()
            is_healthy = await vector_store.health_check()
            latency = (time.time() - start) * 1000
            
            if is_healthy:
                components["vector_store"] = ComponentHealth(
                    status=HealthStatus.HEALTHY,
                    message="Vector store is healthy",
                    latency_ms=latency
                )
            else:
                components["vector_store"] = ComponentHealth(
                    status=HealthStatus.UNHEALTHY,
                    message="Vector store health check failed"
                )
                overall_status = HealthStatus.DEGRADED
        
        except Exception as e:
            components["vector_store"] = ComponentHealth(
                status=HealthStatus.UNHEALTHY,
                message=f"Vector store error: {str(e)}"
            )
            overall_status = HealthStatus.DEGRADED
    
    # Check LLM if requested
    if request.check_llm:
        try:
            start = time.time()
            synthesis_agent = await AgentFactory.get_synthesis_agent()
            is_healthy = await synthesis_agent.health_check()
            latency = (time.time() - start) * 1000
            
            if is_healthy:
                components["llm"] = ComponentHealth(
                    status=HealthStatus.HEALTHY,
                    message="LLM API is healthy",
                    latency_ms=latency
                )
            else:
                components["llm"] = ComponentHealth(
                    status=HealthStatus.UNHEALTHY,
                    message="LLM API health check failed"
                )
                overall_status = HealthStatus.DEGRADED
        
        except Exception as e:
            components["llm"] = ComponentHealth(
                status=HealthStatus.UNHEALTHY,
                message=f"LLM API error: {str(e)}"
            )
            overall_status = HealthStatus.DEGRADED
    
    # Check agents if requested
    if request.check_agents:
        try:
            agent_health = await AgentFactory.health_check_all()
            
            for agent_name, is_healthy in agent_health.items():
                if is_healthy:
                    components[f"agent_{agent_name}"] = ComponentHealth(
                        status=HealthStatus.HEALTHY,
                        message=f"{agent_name} agent is healthy"
                    )
                else:
                    components[f"agent_{agent_name}"] = ComponentHealth(
                        status=HealthStatus.UNHEALTHY,
                        message=f"{agent_name} agent is unhealthy"
                    )
                    overall_status = HealthStatus.DEGRADED
        
        except Exception as e:
            components["agents"] = ComponentHealth(
                status=HealthStatus.UNHEALTHY,
                message=f"Agent health check error: {str(e)}"
            )
            overall_status = HealthStatus.DEGRADED
    
    return HealthCheckResponse(
        status=overall_status,
        version=settings.app_version,
        uptime_seconds=uptime,
        components=components
    )


@router.get("/health/ready")
async def readiness_check():
    """Kubernetes readiness probe"""
    try:
        # Check if critical components are ready
        vector_store = await VectorStoreFactory.get_vector_store()
        is_ready = await vector_store.health_check()
        
        if is_ready:
            return {"status": "ready"}
        else:
            return {"status": "not_ready"}, 503
    
    except Exception:
        return {"status": "not_ready"}, 503


@router.get("/health/live")
async def liveness_check():
    """Kubernetes liveness probe"""
    # Simple check that the process is alive
    return {"status": "alive"}


@router.get("/metrics/system")
async def system_metrics():
    """System resource metrics"""
    cpu_percent = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage('/')
    
    return {
        "cpu": {
            "percent": cpu_percent,
            "count": psutil.cpu_count()
        },
        "memory": {
            "total_mb": memory.total / (1024 * 1024),
            "available_mb": memory.available / (1024 * 1024),
            "percent": memory.percent
        },
        "disk": {
            "total_gb": disk.total / (1024 * 1024 * 1024),
            "used_gb": disk.used / (1024 * 1024 * 1024),
            "percent": disk.percent
        },
        "uptime_seconds": time.time() - START_TIME
    }