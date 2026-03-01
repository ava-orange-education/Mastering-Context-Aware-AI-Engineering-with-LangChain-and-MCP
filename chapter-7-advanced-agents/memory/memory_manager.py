"""
Unified memory management system combining short-term, long-term, and semantic memory.
"""

from typing import Dict, Any, List, Optional
from .short_term_memory import ConversationBufferMemory
from .long_term_memory import EpisodicMemory
from .semantic_memory import SemanticMemory
import logging

logger = logging.getLogger(__name__)


class UnifiedMemoryManager:
    """
    Manages all types of memory for an agent.
    """
    
    def __init__(self, agent_id: str, llm_client, embedding_manager, vector_store):
        """
        Initialize unified memory manager
        
        Args:
            agent_id: Unique agent identifier
            llm_client: LLM client
            embedding_manager: Embedding generator
            vector_store: Vector database
        """
        self.agent_id = agent_id
        
        # Initialize different memory systems
        self.short_term = ConversationBufferMemory(max_messages=10)
        self.episodic = EpisodicMemory(embedding_manager, vector_store, agent_id)
        self.semantic = SemanticMemory(embedding_manager, vector_store)
        
        self.llm = llm_client
    
    def add_interaction(self, user_input: str, agent_response: str,
                       success: bool = True, metadata: Optional[Dict] = None):
        """
        Add interaction to both short-term and long-term memory
        
        Args:
            user_input: User's input
            agent_response: Agent's response
            success: Whether interaction was successful
            metadata: Additional metadata
        """
        # Add to short-term memory
        self.short_term.add_user_message(user_input)
        self.short_term.add_assistant_message(agent_response)
        
        # Store in episodic memory for long-term recall
        experience = {
            'content': f"User: {user_input}\nAgent: {agent_response}",
            'outcome': 'success' if success else 'failure',
            'success': success,
            'type': 'interaction',
            'metadata': metadata or {}
        }
        
        self.episodic.store_experience(experience)
    
    def get_conversation_context(self, include_relevant_experiences: bool = True,
                                num_experiences: int = 3) -> str:
        """
        Get formatted context including short-term and relevant long-term memories
        
        Args:
            include_relevant_experiences: Whether to include past experiences
            num_experiences: Number of past experiences to include
            
        Returns:
            Formatted context string
        """
        context_parts = []
        
        # Add short-term conversation history
        conversation = self.short_term.get_context_string()
        if conversation:
            context_parts.append("Recent conversation:")
            context_parts.append(conversation)
        
        # Add relevant past experiences if requested
        if include_relevant_experiences and conversation:
            # Use last message as query for relevant experiences
            last_messages = self.short_term.get_last_n_messages(2)
            if last_messages:
                query = " ".join([m['content'] for m in last_messages])
                
                relevant_exp = self.episodic.recall_similar_experiences(
                    query=query,
                    top_k=num_experiences
                )
                
                if relevant_exp:
                    context_parts.append("\nRelevant past experiences:")
                    for exp in relevant_exp:
                        context_parts.append(f"- {exp.get('metadata', {}).get('text', '')[:200]}")
        
        return "\n".join(context_parts)
    
    def learn_from_interaction(self, interaction_summary: str):
        """Extract and store key learnings from interaction"""
        # Use LLM to extract facts and concepts
        prompt = f"""Analyze this interaction and extract:
1. Key facts learned
2. Important concepts discussed
3. Useful patterns or strategies

Interaction: {interaction_summary}

Return as structured text."""
        
        try:
            response = self.llm.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}]
            )
            
            learnings = response.content[0].text
            
            # Store in semantic memory
            self.semantic.store_fact(
                fact=learnings,
                category="learned_knowledge",
                metadata={'source': 'interaction', 'agent_id': self.agent_id}
            )
            
            logger.info("Extracted and stored learnings from interaction")
            
        except Exception as e:
            logger.error(f"Failed to extract learnings: {e}")
    
    def recall_relevant_knowledge(self, query: str, top_k: int = 5) -> List[str]:
        """Recall relevant knowledge from semantic memory"""
        results = self.semantic.retrieve_relevant_facts(query, top_k=top_k)
        return [r.get('metadata', {}).get('fact', '') for r in results]
    
    def clear_short_term_memory(self):
        """Clear short-term conversation memory"""
        self.short_term.clear()
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get statistics about memory usage"""
        return {
            'short_term_messages': len(self.short_term.get_messages()),
            'episodic_summary': self.episodic.get_experience_summary(),
            'agent_id': self.agent_id
        }