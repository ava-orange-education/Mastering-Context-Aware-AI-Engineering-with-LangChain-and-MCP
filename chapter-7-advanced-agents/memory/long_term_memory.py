"""
Long-term episodic memory using vector storage.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class EpisodicMemory:
    """
    Long-term memory that stores and retrieves experiences using embeddings.
    """
    
    def __init__(self, embedding_manager, vector_store, agent_id: str):
        """
        Initialize episodic memory
        
        Args:
            embedding_manager: Embedding generator
            vector_store: Vector database
            agent_id: Unique agent identifier
        """
        self.embedding_mgr = embedding_manager
        self.vector_store = vector_store
        self.agent_id = agent_id
        self.collection_name = f"episodic_memory_{agent_id}"
    
    def store_experience(self, experience: Dict[str, Any]):
        """
        Store an experience in long-term memory
        
        Args:
            experience: Dictionary with 'content', 'outcome', 'metadata'
        """
        # Create experience text
        experience_text = f"""Experience: {experience['content']}
Outcome: {experience.get('outcome', 'Unknown')}
Context: {experience.get('context', '')}"""
        
        # Generate embedding
        embedding_result = self.embedding_mgr.embed_text(experience_text)
        
        # Add metadata
        metadata = {
            'agent_id': self.agent_id,
            'timestamp': datetime.now().isoformat(),
            'experience_type': experience.get('type', 'general'),
            'success': experience.get('success', True),
            **experience.get('metadata', {})
        }
        
        # Store in vector database
        self.vector_store.add_vectors(
            vectors=[embedding_result.embedding],
            metadata=[{
                'text': experience_text,
                **metadata
            }],
            ids=[f"{self.agent_id}_{datetime.now().timestamp()}"]
        )
        
        logger.info(f"Stored experience for agent {self.agent_id}")
    
    def recall_similar_experiences(self, query: str, top_k: int = 5,
                                  filter_metadata: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """
        Recall similar past experiences
        
        Args:
            query: Query describing situation
            top_k: Number of experiences to recall
            filter_metadata: Optional metadata filters
            
        Returns:
            List of similar experiences
        """
        # Generate query embedding
        query_embedding = self.embedding_mgr.embed_text(query).embedding
        
        # Add agent_id to filter
        if filter_metadata is None:
            filter_metadata = {}
        filter_metadata['agent_id'] = self.agent_id
        
        # Search vector store
        results = self.vector_store.search(
            query_vector=query_embedding,
            top_k=top_k,
            filter=filter_metadata
        )
        
        return results
    
    def recall_successful_experiences(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Recall only successful experiences"""
        return self.recall_similar_experiences(
            query=query,
            top_k=top_k,
            filter_metadata={'success': True}
        )
    
    def recall_recent_experiences(self, hours: int = 24, top_k: int = 10) -> List[Dict[str, Any]]:
        """Recall experiences from recent time window"""
        from datetime import timedelta
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        cutoff_iso = cutoff_time.isoformat()
        
        # Note: This requires vector store to support timestamp filtering
        # Implementation depends on specific vector store capabilities
        return self.recall_similar_experiences(
            query="recent experiences",
            top_k=top_k,
            filter_metadata={'timestamp': {'$gte': cutoff_iso}}
        )
    
    def get_experience_summary(self, experience_type: Optional[str] = None) -> Dict[str, Any]:
        """Get summary statistics about stored experiences"""
        # This is a simplified version - real implementation would query vector store
        filter_meta = {'agent_id': self.agent_id}
        if experience_type:
            filter_meta['experience_type'] = experience_type
        
        # Query for all experiences (limited)
        results = self.recall_similar_experiences(
            query="all experiences",
            top_k=100,
            filter_metadata=filter_meta
        )
        
        total = len(results)
        successful = sum(1 for r in results if r.get('metadata', {}).get('success', False))
        
        return {
            'total_experiences': total,
            'successful_experiences': successful,
            'success_rate': successful / total if total > 0 else 0,
            'experience_type': experience_type
        }


class ExperienceReplayMemory(EpisodicMemory):
    """Memory that can replay and learn from past experiences"""
    
    def replay_experience(self, experience_id: str) -> Dict[str, Any]:
        """Retrieve and analyze a specific experience"""
        # Retrieve experience
        result = self.vector_store.get_by_id(experience_id)
        
        if not result:
            return {'error': 'Experience not found'}
        
        return {
            'experience': result,
            'lessons': self._extract_lessons(result)
        }
    
    def _extract_lessons(self, experience: Dict[str, Any]) -> List[str]:
        """Extract lessons learned from experience"""
        # This would use LLM to analyze the experience
        # Simplified version here
        lessons = []
        
        if experience.get('metadata', {}).get('success'):
            lessons.append("This approach was successful")
        else:
            lessons.append("This approach failed - consider alternatives")
        
        return lessons