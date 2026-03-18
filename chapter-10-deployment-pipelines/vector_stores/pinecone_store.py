"""
Pinecone vector store implementation
"""

import pinecone
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime

from .base_store import BaseVectorStore, Document, SearchResult
from config.settings import settings

logger = logging.getLogger(__name__)


class PineconeStore(BaseVectorStore):
    """Pinecone vector store implementation"""
    
    def __init__(self):
        self.api_key = settings.pinecone_api_key
        self.environment = settings.pinecone_environment
        self.index_name = settings.pinecone_index_name
        self.index = None
        self._initialized = False
    
    async def initialize(self) -> None:
        """Initialize Pinecone connection"""
        try:
            pinecone.init(
                api_key=self.api_key,
                environment=self.environment
            )
            
            # Check if index exists
            if self.index_name not in pinecone.list_indexes():
                logger.warning(
                    f"Index '{self.index_name}' does not exist. "
                    "Call create_index() to create it."
                )
            else:
                self.index = pinecone.Index(self.index_name)
                logger.info(f"Connected to Pinecone index: {self.index_name}")
            
            self._initialized = True
            
        except Exception as e:
            logger.error(f"Failed to initialize Pinecone: {e}")
            raise
    
    async def create_index(
        self,
        dimension: int,
        metric: str = "cosine",
        pod_type: str = "p1.x1",
        **kwargs
    ) -> None:
        """Create a new Pinecone index"""
        try:
            if self.index_name in pinecone.list_indexes():
                logger.warning(f"Index '{self.index_name}' already exists")
                self.index = pinecone.Index(self.index_name)
                return
            
            pinecone.create_index(
                name=self.index_name,
                dimension=dimension,
                metric=metric,
                pod_type=pod_type
            )
            
            self.index = pinecone.Index(self.index_name)
            logger.info(
                f"Created Pinecone index: {self.index_name} "
                f"(dimension={dimension}, metric={metric})"
            )
            
        except Exception as e:
            logger.error(f"Failed to create Pinecone index: {e}")
            raise
    
    async def upsert(self, documents: List[Document]) -> None:
        """Insert or update documents in Pinecone"""
        if not self._initialized or self.index is None:
            raise RuntimeError("Pinecone store not initialized")
        
        try:
            # Prepare vectors for upsert
            vectors = []
            for doc in documents:
                if doc.embedding is None:
                    raise ValueError(f"Document {doc.id} has no embedding")
                
                vector_data = {
                    "id": doc.id,
                    "values": doc.embedding,
                    "metadata": {
                        "content": doc.content,
                        **(doc.metadata or {})
                    }
                }
                vectors.append(vector_data)
            
            # Upsert in batches of 100
            batch_size = 100
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:i + batch_size]
                self.index.upsert(vectors=batch)
            
            logger.info(f"Upserted {len(documents)} documents to Pinecone")
            
        except Exception as e:
            logger.error(f"Failed to upsert documents: {e}")
            raise
    
    async def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Search for similar documents"""
        if not self._initialized or self.index is None:
            raise RuntimeError("Pinecone store not initialized")
        
        try:
            # Query Pinecone
            results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True,
                filter=filter
            )
            
            # Convert to SearchResult objects
            search_results = []
            for match in results.matches:
                metadata = match.metadata or {}
                content = metadata.pop("content", "")
                
                document = Document(
                    id=match.id,
                    content=content,
                    embedding=None,  # Not returned by default
                    metadata=metadata
                )
                
                search_results.append(
                    SearchResult(document=document, score=match.score)
                )
            
            logger.info(f"Found {len(search_results)} results from Pinecone")
            return search_results
            
        except Exception as e:
            logger.error(f"Failed to search Pinecone: {e}")
            raise
    
    async def delete(self, document_ids: List[str]) -> None:
        """Delete documents from Pinecone"""
        if not self._initialized or self.index is None:
            raise RuntimeError("Pinecone store not initialized")
        
        try:
            self.index.delete(ids=document_ids)
            logger.info(f"Deleted {len(document_ids)} documents from Pinecone")
            
        except Exception as e:
            logger.error(f"Failed to delete documents: {e}")
            raise
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get Pinecone index statistics"""
        if not self._initialized or self.index is None:
            raise RuntimeError("Pinecone store not initialized")
        
        try:
            stats = self.index.describe_index_stats()
            
            return {
                "total_vector_count": stats.total_vector_count,
                "dimension": stats.dimension,
                "index_fullness": stats.index_fullness,
                "namespaces": dict(stats.namespaces) if stats.namespaces else {}
            }
            
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            raise
    
    async def health_check(self) -> bool:
        """Check if Pinecone is healthy"""
        try:
            if not self._initialized or self.index is None:
                return False
            
            # Try to get stats as health check
            await self.get_stats()
            return True
            
        except Exception as e:
            logger.error(f"Pinecone health check failed: {e}")
            return False