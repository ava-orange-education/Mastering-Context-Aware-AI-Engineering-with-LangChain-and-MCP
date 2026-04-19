"""
Permission-Aware Retriever

Retrieves documents while respecting access permissions
"""

from typing import List, Dict, Any, Optional
import logging
import sys
sys.path.append('../..')

from shared.base_rag import BaseRAG, Document, SearchResult
from shared.config import get_settings
from access_control.permission_manager import PermissionManager

logger = logging.getLogger(__name__)
settings = get_settings()


class PermissionAwareRetriever(BaseRAG):
    """
    RAG retriever that enforces document permissions
    """
    
    def __init__(self):
        super().__init__(
            collection_name="enterprise_documents",
            embedding_model="voyage-2"
        )
        
        self.vector_store = None
        self.permission_manager = PermissionManager()
    
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
        
        logger.info("Permission-aware retriever initialized")
    
    async def search(
        self,
        query: str,
        user_id: str,
        user_groups: Optional[List[str]] = None,
        top_k: int = 10,
        sources: Optional[List[str]] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """
        Search with permission filtering
        
        Args:
            query: Search query
            user_id: User performing search
            user_groups: User's group memberships
            top_k: Number of results
            sources: Source systems to search
            filters: Additional metadata filters
        
        Returns:
            List of accessible search results
        """
        
        # Generate embedding
        from ingestion.embedding_pipeline import EmbeddingPipeline
        embedder = EmbeddingPipeline()
        query_embedding = await embedder.generate_embedding(query)
        
        # Build filters
        search_filters = filters or {}
        
        # Add source filter if specified
        if sources and "all" not in sources:
            search_filters["source"] = {"$in": sources}
        
        # Search vector store (get more results for permission filtering)
        initial_results = await self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k * 3,  # Get 3x results to account for permission filtering
            filter=search_filters
        )
        
        # Filter by permissions
        accessible_results = []
        
        for result in initial_results:
            doc_id = result.document.id
            
            # Check if user has access
            has_access = await self.permission_manager.check_access(
                user_id=user_id,
                document_id=doc_id,
                user_groups=user_groups or []
            )
            
            if has_access:
                accessible_results.append(result)
                
                # Stop when we have enough results
                if len(accessible_results) >= top_k:
                    break
        
        logger.info(
            f"Retrieved {len(accessible_results)} accessible documents "
            f"from {len(initial_results)} total results"
        )
        
        return accessible_results
    
    async def upsert(self, documents: List[Document]) -> None:
        """
        Upsert documents with permission metadata
        
        Args:
            documents: Documents to upsert
        """
        
        # Ensure permission metadata is present
        for doc in documents:
            if not doc.metadata:
                doc.metadata = {}
            
            # Add default permissions if not specified
            if "permissions" not in doc.metadata:
                doc.metadata["permissions"] = {
                    "public": False,
                    "groups": [],
                    "users": []
                }
        
        await self.vector_store.upsert(documents)
        
        logger.info(f"Upserted {len(documents)} documents with permissions")
    
    async def delete(self, document_ids: List[str]) -> None:
        """Delete documents"""
        await self.vector_store.delete(document_ids)
    
    async def health_check(self) -> bool:
        """Check if retriever is healthy"""
        return await self.vector_store.health_check()
    
    async def get_accessible_documents(
        self,
        user_id: str,
        user_groups: List[str],
        document_ids: List[str]
    ) -> List[str]:
        """
        Filter document IDs to only accessible ones
        
        Args:
            user_id: User ID
            user_groups: User's groups
            document_ids: Document IDs to filter
        
        Returns:
            List of accessible document IDs
        """
        
        accessible = []
        
        for doc_id in document_ids:
            has_access = await self.permission_manager.check_access(
                user_id=user_id,
                document_id=doc_id,
                user_groups=user_groups
            )
            
            if has_access:
                accessible.append(doc_id)
        
        return accessible