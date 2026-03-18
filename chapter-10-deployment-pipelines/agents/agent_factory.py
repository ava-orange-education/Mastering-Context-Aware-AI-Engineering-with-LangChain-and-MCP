"""
Factory for creating and managing agents
"""

from typing import Optional
import logging

from .retrieval_agent import RetrievalAgent
from .analysis_agent import AnalysisAgent
from .synthesis_agent import SynthesisAgent
from vector_stores.store_factory import VectorStoreFactory

logger = logging.getLogger(__name__)


class AgentFactory:
    """Factory for creating and managing agent instances"""
    
    _retrieval_agent: Optional[RetrievalAgent] = None
    _analysis_agent: Optional[AnalysisAgent] = None
    _synthesis_agent: Optional[SynthesisAgent] = None
    
    @classmethod
    async def get_retrieval_agent(cls) -> RetrievalAgent:
        """Get or create retrieval agent (singleton)"""
        if cls._retrieval_agent is None:
            vector_store = await VectorStoreFactory.get_vector_store()
            cls._retrieval_agent = RetrievalAgent(vector_store)
            logger.info("Created retrieval agent")
        
        return cls._retrieval_agent
    
    @classmethod
    async def get_analysis_agent(cls) -> AnalysisAgent:
        """Get or create analysis agent (singleton)"""
        if cls._analysis_agent is None:
            cls._analysis_agent = AnalysisAgent()
            logger.info("Created analysis agent")
        
        return cls._analysis_agent
    
    @classmethod
    async def get_synthesis_agent(cls) -> SynthesisAgent:
        """Get or create synthesis agent (singleton)"""
        if cls._synthesis_agent is None:
            cls._synthesis_agent = SynthesisAgent()
            logger.info("Created synthesis agent")
        
        return cls._synthesis_agent
    
    @classmethod
    async def initialize_all_agents(cls):
        """Initialize all agents"""
        await cls.get_retrieval_agent()
        await cls.get_analysis_agent()
        await cls.get_synthesis_agent()
        logger.info("All agents initialized")
    
    @classmethod
    async def health_check_all(cls) -> dict:
        """Check health of all agents"""
        results = {}
        
        try:
            retrieval_agent = await cls.get_retrieval_agent()
            results["retrieval"] = await retrieval_agent.health_check()
        except Exception as e:
            logger.error(f"Retrieval agent health check failed: {e}")
            results["retrieval"] = False
        
        try:
            analysis_agent = await cls.get_analysis_agent()
            results["analysis"] = await analysis_agent.health_check()
        except Exception as e:
            logger.error(f"Analysis agent health check failed: {e}")
            results["analysis"] = False
        
        try:
            synthesis_agent = await cls.get_synthesis_agent()
            results["synthesis"] = await synthesis_agent.health_check()
        except Exception as e:
            logger.error(f"Synthesis agent health check failed: {e}")
            results["synthesis"] = False
        
        return results