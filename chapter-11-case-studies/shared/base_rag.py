"""
Base RAG classes for all case studies
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class Document:
    """Document representation"""
    
    id: str
    content: str
    metadata: Optional[Dict[str, Any]] = None
    embedding: Optional[List[float]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "id": self.id,
            "content": self.content,
            "metadata": self.metadata or {},
            "has_embedding": self.embedding is not None
        }


@dataclass
class SearchResult:
    """Search result representation"""
    
    document: Document
    score: float
    rank: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "document": self.document.to_dict(),
            "score": self.score,
            "rank": self.rank
        }


class BaseRAG(ABC):
    """Base class for RAG implementations"""
    
    def __init__(
        self,
        collection_name: str,
        embedding_model: str = "voyage-2"
    ):
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        
        logger.info(f"Initialized RAG: {self.collection_name}")
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize vector store connection"""
        pass
    
    @abstractmethod
    async def upsert(self, documents: List[Document]) -> None:
        """Upsert documents to vector store"""
        pass
    
    @abstractmethod
    async def search(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Search for relevant documents"""
        pass
    
    @abstractmethod
    async def delete(self, document_ids: List[str]) -> None:
        """Delete documents from vector store"""
        pass
    
    async def batch_upsert(
        self,
        documents: List[Document],
        batch_size: int = 100
    ) -> None:
        """Upsert documents in batches"""
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            await self.upsert(batch)
            logger.info(f"Upserted batch {i // batch_size + 1}")
    
    @abstractmethod
    async def health_check(self) -> bool:
        """Check if RAG system is healthy"""
        pass
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get RAG metadata"""
        return {
            "collection_name": self.collection_name,
            "embedding_model": self.embedding_model
        }