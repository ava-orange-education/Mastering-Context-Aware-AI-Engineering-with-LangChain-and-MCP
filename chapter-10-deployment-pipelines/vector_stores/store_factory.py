"""
Factory for creating vector store instances
"""

from typing import Optional
import logging

from .base_store import BaseVectorStore
from .pinecone_store import PineconeStore
from .weaviate_store import WeaviateStore
from config.settings import settings, VectorStoreType

logger = logging.getLogger(__name__)


class VectorStoreFactory:
    """Factory for creating vector store instances"""
    
    _instance: Optional[BaseVectorStore] = None
    
    @classmethod
    async def get_vector_store(
        cls,
        store_type: Optional[VectorStoreType] = None
    ) -> BaseVectorStore:
        """Get or create vector store instance (singleton)"""
        
        if cls._instance is not None:
            return cls._instance
        
        # Use provided type or default from settings
        store_type = store_type or settings.vector_store
        
        # Create appropriate store
        if store_type == VectorStoreType.PINECONE:
            logger.info("Creating Pinecone vector store")
            cls._instance = PineconeStore()
        
        elif store_type == VectorStoreType.WEAVIATE:
            logger.info("Creating Weaviate vector store")
            cls._instance = WeaviateStore()
        
        else:
            raise ValueError(f"Unknown vector store type: {store_type}")
        
        # Initialize the store
        await cls._instance.initialize()
        
        return cls._instance
    
    @classmethod
    async def close(cls):
        """Close vector store connection"""
        if cls._instance:
            await cls._instance.close()
            cls._instance = None