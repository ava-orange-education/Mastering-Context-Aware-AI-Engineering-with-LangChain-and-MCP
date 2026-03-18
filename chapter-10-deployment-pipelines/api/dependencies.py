"""
FastAPI dependencies
"""

from fastapi import Depends, HTTPException, status
from typing import Optional, Dict
import logging

from agents.agent_factory import AgentFactory
from agents import RetrievalAgent, AnalysisAgent, SynthesisAgent
from vector_stores.store_factory import VectorStoreFactory
from vector_stores.base_store import BaseVectorStore
from api.middleware.auth_middleware import optional_auth

logger = logging.getLogger(__name__)


async def get_retrieval_agent() -> RetrievalAgent:
    """Dependency to get retrieval agent"""
    try:
        return await AgentFactory.get_retrieval_agent()
    except Exception as e:
        logger.error(f"Failed to get retrieval agent: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Retrieval agent unavailable"
        )


async def get_analysis_agent() -> AnalysisAgent:
    """Dependency to get analysis agent"""
    try:
        return await AgentFactory.get_analysis_agent()
    except Exception as e:
        logger.error(f"Failed to get analysis agent: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Analysis agent unavailable"
        )


async def get_synthesis_agent() -> SynthesisAgent:
    """Dependency to get synthesis agent"""
    try:
        return await AgentFactory.get_synthesis_agent()
    except Exception as e:
        logger.error(f"Failed to get synthesis agent: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Synthesis agent unavailable"
        )


async def get_vector_store() -> BaseVectorStore:
    """Dependency to get vector store"""
    try:
        return await VectorStoreFactory.get_vector_store()
    except Exception as e:
        logger.error(f"Failed to get vector store: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Vector store unavailable"
        )


async def get_current_user_optional(
    user: Optional[Dict] = Depends(optional_auth)
) -> Optional[Dict]:
    """Dependency for optional user authentication"""
    return user


def verify_api_version(api_version: str = "v1") -> str:
    """Dependency to verify API version"""
    supported_versions = ["v1"]
    
    if api_version not in supported_versions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported API version: {api_version}"
        )
    
    return api_version