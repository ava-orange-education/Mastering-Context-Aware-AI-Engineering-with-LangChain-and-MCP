"""
Manage trust relationships between agents.
"""

from typing import Dict, Set, Optional
from dataclasses import dataclass, field
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TrustRelationship:
    """Trust relationship between agents"""
    agent_a: str
    agent_b: str
    trust_level: float  # 0.0 to 1.0
    bidirectional: bool = True
    metadata: Dict = field(default_factory=dict)


class TrustManager:
    """Manage agent trust relationships"""
    
    def __init__(self):
        """Initialize trust manager"""
        self.trust_relationships: Dict[str, TrustRelationship] = {}
        self.trust_matrix: Dict[str, Dict[str, float]] = {}
    
    def establish_trust(self, agent_a: str, agent_b: str, 
                       trust_level: float = 1.0, bidirectional: bool = True):
        """
        Establish trust relationship
        
        Args:
            agent_a: First agent
            agent_b: Second agent
            trust_level: Trust level (0.0-1.0)
            bidirectional: Whether trust is mutual
        """
        relationship_id = f"{agent_a}:{agent_b}"
        
        relationship = TrustRelationship(
            agent_a=agent_a,
            agent_b=agent_b,
            trust_level=trust_level,
            bidirectional=bidirectional
        )
        
        self.trust_relationships[relationship_id] = relationship
        
        # Update trust matrix
        if agent_a not in self.trust_matrix:
            self.trust_matrix[agent_a] = {}
        self.trust_matrix[agent_a][agent_b] = trust_level
        
        if bidirectional:
            if agent_b not in self.trust_matrix:
                self.trust_matrix[agent_b] = {}
            self.trust_matrix[agent_b][agent_a] = trust_level
        
        logger.info(f"Trust established: {agent_a} <-> {agent_b} (level: {trust_level})")
    
    def get_trust_level(self, agent_from: str, agent_to: str) -> float:
        """
        Get trust level from one agent to another
        
        Args:
            agent_from: Source agent
            agent_to: Target agent
            
        Returns:
            Trust level (0.0 if no trust established)
        """
        if agent_from in self.trust_matrix:
            return self.trust_matrix[agent_from].get(agent_to, 0.0)
        return 0.0
    
    def is_trusted(self, agent_from: str, agent_to: str, 
                   min_trust: float = 0.5) -> bool:
        """
        Check if agent_to is trusted by agent_from
        
        Args:
            agent_from: Source agent
            agent_to: Target agent
            min_trust: Minimum trust threshold
            
        Returns:
            True if trusted above threshold
        """
        trust_level = self.get_trust_level(agent_from, agent_to)
        return trust_level >= min_trust
    
    def get_trusted_agents(self, agent_id: str, min_trust: float = 0.5) -> List[str]:
        """
        Get list of agents trusted by given agent
        
        Args:
            agent_id: Agent to check
            min_trust: Minimum trust threshold
            
        Returns:
            List of trusted agent IDs
        """
        if agent_id not in self.trust_matrix:
            return []
        
        trusted = [
            other_agent 
            for other_agent, trust in self.trust_matrix[agent_id].items()
            if trust >= min_trust
        ]
        
        return trusted
    
    def revoke_trust(self, agent_a: str, agent_b: str):
        """Revoke trust relationship"""
        relationship_id = f"{agent_a}:{agent_b}"
        
        if relationship_id in self.trust_relationships:
            del self.trust_relationships[relationship_id]
        
        if agent_a in self.trust_matrix and agent_b in self.trust_matrix[agent_a]:
            del self.trust_matrix[agent_a][agent_b]
        
        logger.info(f"Trust revoked: {agent_a} -> {agent_b}")