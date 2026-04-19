"""
Knowledge Base Retriever

Retrieves DevOps knowledge, best practices, and documentation
"""

from typing import List, Dict, Any, Optional
import logging
import sys
sys.path.append('../..')

from shared.base_rag import BaseRAG, Document, SearchResult

logger = logging.getLogger(__name__)


class KnowledgeBaseRetriever(BaseRAG):
    """
    Retrieves DevOps knowledge and best practices
    """
    
    def __init__(self):
        super().__init__(
            collection_name="devops_knowledge",
            embedding_model="voyage-2"
        )
        
        self.vector_store = None
    
    async def initialize(self) -> None:
        """Initialize vector store"""
        
        from shared.config import get_settings
        settings = get_settings()
        
        if settings.vector_store == "pinecone":
            from vector_stores.pinecone_store import PineconeStore
            self.vector_store = PineconeStore()
        else:
            from vector_stores.weaviate_store import WeaviateStore
            self.vector_store = WeaviateStore()
        
        await self.vector_store.initialize()
        
        logger.info("Knowledge base retriever initialized")
    
    async def search_knowledge(
        self,
        query: str,
        category: Optional[str] = None,
        top_k: int = 5
    ) -> List[SearchResult]:
        """
        Search knowledge base
        
        Args:
            query: Search query
            category: Knowledge category (troubleshooting, best_practices, etc.)
            top_k: Number of results
        
        Returns:
            List of relevant knowledge articles
        """
        
        from ingestion.embedding_pipeline import EmbeddingPipeline
        embedder = EmbeddingPipeline()
        
        query_embedding = await embedder.generate_embedding(query)
        
        # Build filters
        filters = {}
        if category:
            filters["category"] = category
        
        results = await self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k,
            filter=filters if filters else None
        )
        
        logger.info(f"Found {len(results)} knowledge articles for: {query}")
        
        return results
    
    async def get_best_practices(
        self,
        topic: str,
        top_k: int = 3
    ) -> List[SearchResult]:
        """
        Get best practices for a topic
        
        Args:
            topic: Topic (kubernetes, monitoring, security, etc.)
            top_k: Number of results
        
        Returns:
            Best practices
        """
        
        query = f"{topic} best practices"
        
        return await self.search_knowledge(
            query=query,
            category="best_practices",
            top_k=top_k
        )
    
    async def get_troubleshooting_guide(
        self,
        problem: str,
        component: Optional[str] = None
    ) -> List[SearchResult]:
        """
        Get troubleshooting guides
        
        Args:
            problem: Problem description
            component: Optional component name
        
        Returns:
            Troubleshooting guides
        """
        
        query = f"troubleshooting {problem}"
        if component:
            query += f" {component}"
        
        return await self.search_knowledge(
            query=query,
            category="troubleshooting",
            top_k=5
        )
    
    async def store_knowledge(
        self,
        title: str,
        content: str,
        category: str,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Store knowledge article
        
        Args:
            title: Article title
            content: Article content
            category: Category
            tags: Optional tags
            metadata: Additional metadata
        """
        
        # Generate embedding
        from ingestion.embedding_pipeline import EmbeddingPipeline
        embedder = EmbeddingPipeline()
        
        full_content = f"{title}\n\n{content}"
        embedding = await embedder.generate_embedding(full_content)
        
        # Create document
        doc_metadata = {
            "title": title,
            "category": category,
            "tags": tags or [],
            **(metadata or {})
        }
        
        doc = Document(
            id=f"kb_{title.lower().replace(' ', '_')}",
            content=full_content,
            metadata=doc_metadata,
            embedding=embedding
        )
        
        # Store
        await self.vector_store.upsert([doc])
        
        logger.info(f"Stored knowledge article: {title}")
    
    async def get_documentation(
        self,
        tool: str,
        feature: Optional[str] = None
    ) -> List[SearchResult]:
        """
        Get documentation for a tool or feature
        
        Args:
            tool: Tool name (kubernetes, prometheus, etc.)
            feature: Optional specific feature
        
        Returns:
            Documentation
        """
        
        query = f"{tool} documentation"
        if feature:
            query += f" {feature}"
        
        return await self.search_knowledge(
            query=query,
            category="documentation",
            top_k=3
        )
    
    async def get_common_errors(
        self,
        error_message: str
    ) -> List[SearchResult]:
        """
        Get information about common errors
        
        Args:
            error_message: Error message
        
        Returns:
            Error explanations and solutions
        """
        
        query = f"error: {error_message}"
        
        return await self.search_knowledge(
            query=query,
            category="errors",
            top_k=5
        )
    
    async def upsert(self, documents: List[Document]) -> None:
        """Upsert documents"""
        await self.vector_store.upsert(documents)
    
    async def delete(self, document_ids: List[str]) -> None:
        """Delete documents"""
        await self.vector_store.delete(document_ids)
    
    async def health_check(self) -> bool:
        """Check if retriever is healthy"""
        return await self.vector_store.health_check()
    
    async def populate_default_knowledge(self) -> None:
        """Populate knowledge base with default content"""
        
        # Default best practices
        best_practices = [
            {
                "title": "Kubernetes Pod Health Checks",
                "content": """Always configure liveness and readiness probes for your pods.
                
Liveness probes detect if a container is running. If it fails, Kubernetes restarts the container.
Readiness probes detect if a container is ready to serve traffic.

Best practices:
- Use HTTP endpoints for health checks when possible
- Set appropriate timeout and period values
- Don't make health checks dependent on external services
- Use startup probes for slow-starting containers""",
                "category": "best_practices",
                "tags": ["kubernetes", "health_checks", "reliability"]
            },
            {
                "title": "High Availability Database Configuration",
                "content": """Configure databases for high availability:
                
- Use replication (master-slave or master-master)
- Implement automatic failover
- Regular backups with tested restore procedures
- Monitor replication lag
- Use connection pooling
- Implement circuit breakers in application code""",
                "category": "best_practices",
                "tags": ["database", "high_availability", "reliability"]
            },
            {
                "title": "Monitoring Alert Fatigue Prevention",
                "content": """Prevent alert fatigue:
                
- Alert only on actionable conditions
- Use appropriate severity levels
- Implement alert aggregation
- Set up escalation policies
- Regular alert review and cleanup
- Use SLOs to determine what to alert on""",
                "category": "best_practices",
                "tags": ["monitoring", "alerting", "sre"]
            }
        ]
        
        # Troubleshooting guides
        troubleshooting = [
            {
                "title": "High CPU Usage Troubleshooting",
                "content": """Steps to troubleshoot high CPU usage:
                
1. Identify the process using most CPU (top, htop, or container stats)
2. Check for CPU-intensive operations in logs
3. Review recent code deployments
4. Check for infinite loops or recursive calls
5. Examine database query performance
6. Look for external API calls without timeouts
7. Consider scaling horizontally if legitimate load increase""",
                "category": "troubleshooting",
                "tags": ["cpu", "performance", "debugging"]
            },
            {
                "title": "Memory Leak Investigation",
                "content": """Investigating memory leaks:
                
1. Monitor memory usage over time
2. Take heap dumps at different points
3. Use memory profilers (pprof, jmap, etc.)
4. Check for unclosed connections or file handles
5. Review object lifecycle management
6. Look for growing caches without eviction
7. Check for event listeners that aren't removed""",
                "category": "troubleshooting",
                "tags": ["memory", "performance", "debugging"]
            }
        ]
        
        # Store all knowledge
        for article in best_practices + troubleshooting:
            await self.store_knowledge(
                title=article["title"],
                content=article["content"],
                category=article["category"],
                tags=article["tags"]
            )
        
        logger.info("Populated knowledge base with default content")