"""
Base agent class for all case studies
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime
import logging
import anthropic

logger = logging.getLogger(__name__)


@dataclass
class AgentResponse:
    """Standard agent response format"""
    
    content: str
    agent_name: str
    timestamp: datetime
    metadata: Optional[Dict[str, Any]] = None
    confidence: Optional[float] = None
    sources: Optional[List[str]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "content": self.content,
            "agent_name": self.agent_name,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata or {},
            "confidence": self.confidence,
            "sources": self.sources or []
        }


class BaseAgent(ABC):
    """Base class for all agents across case studies"""
    
    def __init__(
        self,
        name: str,
        model: str = "claude-sonnet-4-20250514",
        temperature: float = 0.7,
        max_tokens: int = 4000
    ):
        self.name = name
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = anthropic.Anthropic()
        
        logger.info(f"Initialized agent: {self.name}")
    
    @abstractmethod
    async def process(self, input_data: Dict[str, Any]) -> AgentResponse:
        """
        Process input and return agent response
        Must be implemented by subclasses
        """
        pass
    
    async def _call_llm(
        self,
        messages: List[Dict[str, str]],
        system: Optional[str] = None
    ) -> str:
        """Call Anthropic LLM"""
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system=system or self._get_system_prompt(),
                messages=messages
            )
            
            return response.content[0].text
        
        except Exception as e:
            logger.error(f"LLM call failed for {self.name}: {e}")
            raise
    
    @abstractmethod
    def _get_system_prompt(self) -> str:
        """Get system prompt for this agent"""
        pass
    
    async def health_check(self) -> bool:
        """Check if agent is healthy"""
        try:
            # Simple health check with minimal LLM call
            response = await self._call_llm(
                messages=[{"role": "user", "content": "ping"}],
                system="Respond with 'pong'"
            )
            return "pong" in response.lower()
        
        except Exception as e:
            logger.error(f"Health check failed for {self.name}: {e}")
            return False
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get agent metadata"""
        return {
            "name": self.name,
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }