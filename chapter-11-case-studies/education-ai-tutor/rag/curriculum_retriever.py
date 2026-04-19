"""
Curriculum Retriever

Retrieves educational content from curriculum database
"""

from typing import List, Dict, Any, Optional
import logging
import sys
sys.path.append('../..')

from shared.base_rag import BaseRAG, Document, SearchResult
from shared.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class CurriculumRetriever(BaseRAG):
    """
    RAG retriever for educational curriculum content
    """
    
    def __init__(self):
        super().__init__(
            collection_name="educational_curriculum",
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
        
        logger.info("Curriculum retriever initialized")
    
    async def search(
        self,
        topic: str,
        grade_level: Optional[str] = None,
        subject: Optional[str] = None,
        content_types: Optional[List[str]] = None,
        difficulty: Optional[str] = None,
        top_k: int = 10
    ) -> List[SearchResult]:
        """
        Search curriculum content
        
        Args:
            topic: Topic to search for
            grade_level: Grade level filter
            subject: Subject filter (math, science, etc.)
            content_types: Types of content (lesson, example, exercise)
            difficulty: Difficulty level
            top_k: Number of results
        
        Returns:
            List of search results
        """
        
        # Generate embedding
        from ingestion.embedding_pipeline import EmbeddingPipeline
        embedder = EmbeddingPipeline()
        query_embedding = await embedder.generate_embedding(topic)
        
        # Build filters
        filters = {}
        
        if grade_level:
            filters["grade_level"] = grade_level
        
        if subject:
            filters["subject"] = subject
        
        if content_types:
            filters["content_type"] = {"$in": content_types}
        
        if difficulty:
            filters["difficulty"] = difficulty
        
        # Search vector store
        results = await self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k,
            filter=filters
        )
        
        logger.info(
            f"Retrieved {len(results)} curriculum items for topic: {topic}"
        )
        
        return results
    
    async def get_lesson_content(
        self,
        topic: str,
        grade_level: str,
        subject: str
    ) -> Optional[Document]:
        """
        Get main lesson content for a topic
        
        Args:
            topic: Topic name
            grade_level: Grade level
            subject: Subject area
        
        Returns:
            Lesson document or None
        """
        
        results = await self.search(
            topic=topic,
            grade_level=grade_level,
            subject=subject,
            content_types=["lesson"],
            top_k=1
        )
        
        return results[0].document if results else None
    
    async def get_examples(
        self,
        topic: str,
        grade_level: str,
        subject: str,
        count: int = 5
    ) -> List[Document]:
        """
        Get example problems for a topic
        
        Args:
            topic: Topic name
            grade_level: Grade level
            subject: Subject area
            count: Number of examples
        
        Returns:
            List of example documents
        """
        
        results = await self.search(
            topic=topic,
            grade_level=grade_level,
            subject=subject,
            content_types=["example"],
            top_k=count
        )
        
        return [r.document for r in results]
    
    async def get_exercises(
        self,
        topic: str,
        grade_level: str,
        subject: str,
        difficulty: str,
        count: int = 10
    ) -> List[Document]:
        """
        Get practice exercises
        
        Args:
            topic: Topic name
            grade_level: Grade level
            subject: Subject area
            difficulty: Difficulty level
            count: Number of exercises
        
        Returns:
            List of exercise documents
        """
        
        results = await self.search(
            topic=topic,
            grade_level=grade_level,
            subject=subject,
            content_types=["exercise"],
            difficulty=difficulty,
            top_k=count
        )
        
        return [r.document for r in results]
    
    async def upsert(self, documents: List[Document]) -> None:
        """
        Upsert curriculum documents
        
        Args:
            documents: Documents to upsert
        """
        
        # Validate educational documents
        for doc in documents:
            if not doc.metadata:
                doc.metadata = {}
            
            # Ensure required metadata
            if "subject" not in doc.metadata:
                logger.warning(f"Document {doc.id} missing subject metadata")
            
            if "grade_level" not in doc.metadata:
                logger.warning(f"Document {doc.id} missing grade_level metadata")
        
        await self.vector_store.upsert(documents)
        
        logger.info(f"Upserted {len(documents)} curriculum documents")
    
    async def delete(self, document_ids: List[str]) -> None:
        """Delete documents"""
        await self.vector_store.delete(document_ids)
    
    async def health_check(self) -> bool:
        """Check if retriever is healthy"""
        return await self.vector_store.health_check()