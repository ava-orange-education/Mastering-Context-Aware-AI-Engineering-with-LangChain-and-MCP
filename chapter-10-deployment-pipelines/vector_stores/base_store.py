"""
Abstract base class for vector stores
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class Document:
    """Document representation"""
    id: str
    content: str
    embedding: Optional[List[float]] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class SearchResult:
    """Search result representation"""
    document: Document
    score: float


class BaseVectorStore(ABC):
    """Abstract base class for vector store implementations"""
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the vector store connection"""
        pass
    
    @abstractmethod
    async def create_index(self, dimension: int, **kwargs) -> None:
        """Create a new index/collection"""
        pass
    
    @abstractmethod
    async def upsert(self, documents: List[Document]) -> None:
        """Insert or update documents"""
        pass
    
    @abstractmethod
    async def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Search for similar documents"""
        pass
    
    @abstractmethod
    async def delete(self, document_ids: List[str]) -> None:
        """Delete documents by ID"""
        pass
    
    @abstractmethod
    async def get_stats(self) -> Dict[str, Any]:
        """Get index statistics"""
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """Check if vector store is healthy"""
        pass
    
    async def close(self) -> None:
        """Close connections and cleanup"""
        pass