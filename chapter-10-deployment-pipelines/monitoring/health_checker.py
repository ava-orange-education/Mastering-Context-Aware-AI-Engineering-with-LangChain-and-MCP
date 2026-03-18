"""
Health check utilities
"""

import asyncio
from typing import Dict, List
import logging

from agents.agent_factory import AgentFactory
from vector_stores.store_factory import VectorStoreFactory

logger = logging.getLogger(__name__)


class HealthChecker:
    """Utilities for checking system health"""
    
    @staticmethod
    async def check_all_components() -> Dict[str, bool]:
        """Check health of all system components"""
        results = {}
        
        # Check vector store
        try:
            vector_store = await VectorStoreFactory.get_vector_store()
            results["vector_store"] = await vector_store.health_check()
        except Exception as e:
            logger.error(f"Vector store health check failed: {e}")
            results["vector_store"] = False
        
        # Check agents
        try:
            agent_health = await AgentFactory.health_check_all()
            results.update({
                f"agent_{name}": status
                for name, status in agent_health.items()
            })
        except Exception as e:
            logger.error(f"Agent health check failed: {e}")
            results["agents"] = False
        
        return results
    
    @staticmethod
    def is_system_healthy(component_health: Dict[str, bool]) -> bool:
        """Determine if system is healthy based on component health"""
        # System is healthy if all critical components are healthy
        critical_components = ["vector_store", "agent_retrieval", "agent_synthesis"]
        
        return all(
            component_health.get(comp, False)
            for comp in critical_components
        )