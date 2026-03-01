"""
Semantic memory for storing and retrieving factual knowledge.
"""

from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)


class SemanticMemory:
    """
    Semantic memory stores factual knowledge and concepts.
    """
    
    def __init__(self, embedding_manager, vector_store):
        """
        Initialize semantic memory
        
        Args:
            embedding_manager: Embedding generator
            vector_store: Vector database for knowledge storage
        """
        self.embedding_mgr = embedding_manager
        self.vector_store = vector_store
        self.knowledge_base: Dict[str, Any] = {}
    
    def store_fact(self, fact: str, category: str = "general", 
                   metadata: Optional[Dict] = None):
        """
        Store a fact in semantic memory
        
        Args:
            fact: The factual statement
            category: Category of knowledge
            metadata: Additional metadata
        """
        # Generate embedding
        embedding = self.embedding_mgr.embed_text(fact).embedding
        
        # Prepare metadata
        meta = {
            'fact': fact,
            'category': category,
            'type': 'semantic_fact',
            **(metadata or {})
        }
        
        # Store in vector database
        fact_id = f"fact_{hash(fact)}"
        self.vector_store.add_vectors(
            vectors=[embedding],
            metadata=[meta],
            ids=[fact_id]
        )
        
        # Also store in local knowledge base for quick access
        self.knowledge_base[fact_id] = meta
        
        logger.info(f"Stored fact in category: {category}")
    
    def retrieve_relevant_facts(self, query: str, top_k: int = 5,
                               category: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Retrieve facts relevant to query
        
        Args:
            query: Query text
            top_k: Number of facts to retrieve
            category: Optional category filter
            
        Returns:
            List of relevant facts
        """
        # Generate query embedding
        query_embedding = self.embedding_mgr.embed_text(query).embedding
        
        # Build filter
        filter_dict = {'type': 'semantic_fact'}
        if category:
            filter_dict['category'] = category
        
        # Search
        results = self.vector_store.search(
            query_vector=query_embedding,
            top_k=top_k,
            filter=filter_dict
        )
        
        return results
    
    def store_concept(self, concept_name: str, definition: str,
                     examples: Optional[List[str]] = None,
                     related_concepts: Optional[List[str]] = None):
        """Store a concept with its definition and relationships"""
        concept_text = f"""Concept: {concept_name}
Definition: {definition}"""
        
        if examples:
            concept_text += f"\nExamples: {', '.join(examples)}"
        
        if related_concepts:
            concept_text += f"\nRelated: {', '.join(related_concepts)}"
        
        self.store_fact(
            fact=concept_text,
            category="concept",
            metadata={
                'concept_name': concept_name,
                'has_examples': bool(examples),
                'has_relations': bool(related_concepts)
            }
        )
    
    def get_concept(self, concept_name: str) -> Optional[Dict[str, Any]]:
        """Retrieve concept by name"""
        results = self.retrieve_relevant_facts(
            query=f"concept: {concept_name}",
            top_k=1,
            category="concept"
        )
        
        if results:
            return results[0]
        return None
    
    def update_fact(self, fact_id: str, new_fact: str):
        """Update an existing fact"""
        # Delete old fact
        self.vector_store.delete([fact_id])
        
        # Store new version
        self.store_fact(new_fact)
    
    def forget_facts(self, category: str):
        """Remove all facts from a category"""
        # This requires vector store support for bulk deletion by metadata
        # Implementation depends on specific vector store
        logger.info(f"Forgetting facts in category: {category}")
        
        # Remove from local knowledge base
        self.knowledge_base = {
            k: v for k, v in self.knowledge_base.items()
            if v.get('category') != category
        }