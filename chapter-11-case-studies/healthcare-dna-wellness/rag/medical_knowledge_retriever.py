"""
Medical Knowledge Retriever

RAG system specialized for medical literature and clinical guidelines
"""

from typing import List, Dict, Any, Optional
import logging
import sys
sys.path.append('../..')

from shared.base_rag import BaseRAG, Document, SearchResult
from shared.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class MedicalKnowledgeRetriever(BaseRAG):
    """
    Specialized RAG for medical knowledge retrieval
    """
    
    def __init__(self):
        super().__init__(
            collection_name="medical_guidelines",
            embedding_model="voyage-2"
        )
        
        self.vector_store = None
    
    async def initialize(self) -> None:
        """Initialize vector store connection"""
        
        # Initialize based on configured vector store
        if settings.vector_store == "pinecone":
            from vector_stores.pinecone_store import PineconeStore
            self.vector_store = PineconeStore()
        else:
            from vector_stores.weaviate_store import WeaviateStore
            self.vector_store = WeaviateStore()
        
        await self.vector_store.initialize()
        
        logger.info("Medical knowledge retriever initialized")
    
    async def upsert(self, documents: List[Document]) -> None:
        """
        Upsert medical documents with validation
        
        Args:
            documents: List of medical documents
        """
        
        # Validate medical documents
        validated_documents = []
        
        for doc in documents:
            if self._validate_medical_document(doc):
                # Add medical-specific metadata
                doc.metadata = doc.metadata or {}
                doc.metadata["document_type"] = doc.metadata.get("document_type", "clinical_guideline")
                doc.metadata["indexed_at"] = datetime.utcnow().isoformat()
                
                validated_documents.append(doc)
            else:
                logger.warning(f"Document {doc.id} failed validation")
        
        await self.vector_store.upsert(validated_documents)
        
        logger.info(f"Upserted {len(validated_documents)} medical documents")
    
    async def search(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """
        Search medical knowledge base
        
        Args:
            query: Search query
            top_k: Number of results
            filters: Metadata filters
        
        Returns:
            List of search results
        """
        
        # Generate embedding for query
        from .embedding_pipeline import EmbeddingPipeline
        embedder = EmbeddingPipeline()
        
        query_embedding = await embedder.generate_embedding(query)
        
        # Search vector store
        results = await self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k,
            filter=filters
        )
        
        # Post-process results
        processed_results = self._post_process_results(results)
        
        logger.info(f"Retrieved {len(processed_results)} results for query")
        
        return processed_results
    
    async def search_by_variant(
        self,
        gene: str,
        variant: str,
        top_k: int = 5
    ) -> List[SearchResult]:
        """
        Search guidelines specific to genetic variant
        
        Args:
            gene: Gene name
            variant: Variant identifier
            top_k: Number of results
        
        Returns:
            List of search results
        """
        
        # Construct variant-specific query
        query = f"{gene} {variant} clinical guidelines management recommendations"
        
        # Add metadata filter for genetic content
        filters = {
            "content_type": "genetic_guideline"
        }
        
        return await self.search(query, top_k, filters)
    
    async def search_by_condition(
        self,
        condition: str,
        top_k: int = 5,
        evidence_level: Optional[str] = None
    ) -> List[SearchResult]:
        """
        Search guidelines for medical condition
        
        Args:
            condition: Medical condition
            top_k: Number of results
            evidence_level: Filter by evidence level (high, moderate, low)
        
        Returns:
            List of search results
        """
        
        query = f"{condition} clinical guidelines evidence-based management"
        
        filters = {}
        if evidence_level:
            filters["evidence_level"] = evidence_level
        
        return await self.search(query, top_k, filters)
    
    async def delete(self, document_ids: List[str]) -> None:
        """Delete documents from vector store"""
        await self.vector_store.delete(document_ids)
    
    async def health_check(self) -> bool:
        """Check if RAG system is healthy"""
        return await self.vector_store.health_check()
    
    def _validate_medical_document(self, doc: Document) -> bool:
        """
        Validate medical document has required fields
        
        Args:
            doc: Document to validate
        
        Returns:
            True if valid
        """
        
        # Check required fields
        if not doc.content or len(doc.content.strip()) == 0:
            return False
        
        if not doc.metadata:
            logger.warning(f"Document {doc.id} missing metadata")
            return False
        
        # Check for source citation
        if "source" not in doc.metadata:
            logger.warning(f"Document {doc.id} missing source")
            return False
        
        return True
    
    def _post_process_results(
        self,
        results: List[SearchResult]
    ) -> List[SearchResult]:
        """
        Post-process search results
        - Filter low-quality results
        - Add relevance annotations
        """
        
        # Filter results below threshold
        threshold = 0.7
        filtered_results = [
            r for r in results
            if r.score >= threshold
        ]
        
        # Add ranking
        for i, result in enumerate(filtered_results):
            result.rank = i + 1
        
        return filtered_results