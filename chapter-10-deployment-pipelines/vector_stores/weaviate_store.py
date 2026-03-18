"""
Weaviate vector store implementation
"""

import weaviate
from weaviate.classes.config import Configure, Property, DataType
from typing import List, Dict, Any, Optional
import logging

from .base_store import BaseVectorStore, Document, SearchResult
from config.settings import settings

logger = logging.getLogger(__name__)


class WeaviateStore(BaseVectorStore):
    """Weaviate vector store implementation"""
    
    def __init__(self):
        self.url = settings.weaviate_url
        self.api_key = settings.weaviate_api_key
        self.class_name = settings.weaviate_class_name
        self.client = None
        self._initialized = False
    
    async def initialize(self) -> None:
        """Initialize Weaviate connection"""
        try:
            # Create Weaviate client
            auth_config = weaviate.AuthApiKey(api_key=self.api_key)
            
            self.client = weaviate.Client(
                url=self.url,
                auth_client_secret=auth_config
            )
            
            # Test connection
            if self.client.is_ready():
                logger.info(f"Connected to Weaviate at {self.url}")
                self._initialized = True
            else:
                raise RuntimeError("Weaviate is not ready")
            
        except Exception as e:
            logger.error(f"Failed to initialize Weaviate: {e}")
            raise
    
    async def create_index(
        self,
        dimension: int,
        vectorizer: str = "none",
        **kwargs
    ) -> None:
        """Create a new Weaviate class (collection)"""
        if not self._initialized or self.client is None:
            raise RuntimeError("Weaviate store not initialized")
        
        try:
            # Check if class exists
            schema = self.client.schema.get()
            existing_classes = [c["class"] for c in schema.get("classes", [])]
            
            if self.class_name in existing_classes:
                logger.warning(f"Class '{self.class_name}' already exists")
                return
            
            # Define class schema
            class_obj = {
                "class": self.class_name,
                "description": "Document storage for RAG system",
                "vectorizer": vectorizer,
                "properties": [
                    {
                        "name": "content",
                        "dataType": ["text"],
                        "description": "Document content"
                    },
                    {
                        "name": "metadata",
                        "dataType": ["object"],
                        "description": "Document metadata"
                    }
                ]
            }
            
            self.client.schema.create_class(class_obj)
            logger.info(f"Created Weaviate class: {self.class_name}")
            
        except Exception as e:
            logger.error(f"Failed to create Weaviate class: {e}")
            raise
    
    async def upsert(self, documents: List[Document]) -> None:
        """Insert or update documents in Weaviate"""
        if not self._initialized or self.client is None:
            raise RuntimeError("Weaviate store not initialized")
        
        try:
            # Use batch import for efficiency
            with self.client.batch as batch:
                batch.batch_size = 100
                
                for doc in documents:
                    if doc.embedding is None:
                        raise ValueError(f"Document {doc.id} has no embedding")
                    
                    data_object = {
                        "content": doc.content,
                        "metadata": doc.metadata or {}
                    }
                    
                    batch.add_data_object(
                        data_object=data_object,
                        class_name=self.class_name,
                        uuid=doc.id,
                        vector=doc.embedding
                    )
            
            logger.info(f"Upserted {len(documents)} documents to Weaviate")
            
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
        if not self._initialized or self.client is None:
            raise RuntimeError("Weaviate store not initialized")
        
        try:
            # Build query
            query = (
                self.client.query
                .get(self.class_name, ["content", "metadata"])
                .with_near_vector({"vector": query_embedding})
                .with_limit(top_k)
                .with_additional(["id", "distance"])
            )
            
            # Add filter if provided
            if filter:
                query = query.with_where(filter)
            
            # Execute query
            results = query.do()
            
            # Convert to SearchResult objects
            search_results = []
            objects = results.get("data", {}).get("Get", {}).get(self.class_name, [])
            
            for obj in objects:
                document = Document(
                    id=obj["_additional"]["id"],
                    content=obj["content"],
                    embedding=None,
                    metadata=obj.get("metadata", {})
                )
                
                # Convert distance to similarity score (closer to 1 is better)
                distance = obj["_additional"]["distance"]
                score = 1 / (1 + distance)
                
                search_results.append(
                    SearchResult(document=document, score=score)
                )
            
            logger.info(f"Found {len(search_results)} results from Weaviate")
            return search_results
            
        except Exception as e:
            logger.error(f"Failed to search Weaviate: {e}")
            raise
    
    async def delete(self, document_ids: List[str]) -> None:
        """Delete documents from Weaviate"""
        if not self._initialized or self.client is None:
            raise RuntimeError("Weaviate store not initialized")
        
        try:
            for doc_id in document_ids:
                self.client.data_object.delete(
                    uuid=doc_id,
                    class_name=self.class_name
                )
            
            logger.info(f"Deleted {len(document_ids)} documents from Weaviate")
            
        except Exception as e:
            logger.error(f"Failed to delete documents: {e}")
            raise
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get Weaviate collection statistics"""
        if not self._initialized or self.client is None:
            raise RuntimeError("Weaviate store not initialized")
        
        try:
            # Get object count
            result = (
                self.client.query
                .aggregate(self.class_name)
                .with_meta_count()
                .do()
            )
            
            count = result.get("data", {}).get("Aggregate", {}).get(
                self.class_name, [{}]
            )[0].get("meta", {}).get("count", 0)
            
            return {
                "total_objects": count,
                "class_name": self.class_name
            }
            
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            raise
    
    async def health_check(self) -> bool:
        """Check if Weaviate is healthy"""
        try:
            if not self._initialized or self.client is None:
                return False
            
            return self.client.is_ready()
            
        except Exception as e:
            logger.error(f"Weaviate health check failed: {e}")
            return False
    
    async def close(self) -> None:
        """Close Weaviate connection"""
        if self.client:
            # Weaviate client doesn't require explicit closing
            self.client = None
            logger.info("Closed Weaviate connection")