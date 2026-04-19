"""
Runbook Retriever

Retrieves relevant runbooks and procedures
"""

from typing import List, Dict, Any, Optional
import logging
import sys
sys.path.append('../..')

from shared.base_rag import BaseRAG, Document, SearchResult

logger = logging.getLogger(__name__)


class RunbookRetriever(BaseRAG):
    """
    Retrieves runbooks and operational procedures
    """
    
    def __init__(self):
        super().__init__(
            collection_name="runbooks",
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
        
        logger.info("Runbook retriever initialized")
    
    async def search_runbooks(
        self,
        query: str,
        incident_type: Optional[str] = None,
        component: Optional[str] = None,
        top_k: int = 3
    ) -> List[SearchResult]:
        """
        Search for relevant runbooks
        
        Args:
            query: Search query
            incident_type: Type of incident
            component: Affected component
            top_k: Number of results
        
        Returns:
            List of relevant runbooks
        """
        
        # Generate embedding
        from ingestion.embedding_pipeline import EmbeddingPipeline
        embedder = EmbeddingPipeline()
        query_embedding = await embedder.generate_embedding(query)
        
        # Build filters
        filters = {}
        
        if incident_type:
            filters["incident_type"] = incident_type
        
        if component:
            filters["component"] = component
        
        # Search
        results = await self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k,
            filter=filters if filters else None
        )
        
        logger.info(f"Found {len(results)} relevant runbooks")
        
        return results
    
    async def get_runbook(
        self,
        runbook_id: str
    ) -> Optional[Document]:
        """Get specific runbook by ID"""
        
        results = await self.vector_store.get_by_ids([runbook_id])
        
        return results[0] if results else None
    
    async def store_runbook(
        self,
        runbook: Dict[str, Any]
    ) -> None:
        """
        Store runbook in knowledge base
        
        Args:
            runbook: Runbook details
        """
        
        # Build runbook content
        content_parts = [
            f"Runbook: {runbook.get('title', 'Unknown')}",
            f"Purpose: {runbook.get('purpose', '')}",
        ]
        
        if "triggers" in runbook:
            content_parts.append(f"Triggers: {runbook['triggers']}")
        
        if "symptoms" in runbook:
            symptoms = ", ".join(runbook["symptoms"])
            content_parts.append(f"Symptoms: {symptoms}")
        
        if "diagnosis_steps" in runbook:
            content_parts.append("Diagnosis:")
            for step in runbook["diagnosis_steps"]:
                content_parts.append(f"  - {step}")
        
        if "remediation_steps" in runbook:
            content_parts.append("Remediation:")
            for step in runbook["remediation_steps"]:
                content_parts.append(f"  - {step}")
        
        if "prevention" in runbook:
            content_parts.append("Prevention:")
            for item in runbook["prevention"]:
                content_parts.append(f"  - {item}")
        
        content = "\n".join(content_parts)
        
        # Generate embedding
        from ingestion.embedding_pipeline import EmbeddingPipeline
        embedder = EmbeddingPipeline()
        embedding = await embedder.generate_embedding(content)
        
        # Create document
        doc = Document(
            id=runbook.get("id", f"runbook_{runbook.get('title', 'unknown')}"),
            content=content,
            metadata={
                "title": runbook.get("title"),
                "incident_type": runbook.get("incident_type"),
                "component": runbook.get("component"),
                "severity": runbook.get("severity"),
                "tags": runbook.get("tags", [])
            },
            embedding=embedding
        )
        
        # Store
        await self.vector_store.upsert([doc])
        
        logger.info(f"Stored runbook: {doc.id}")
    
    def parse_runbook(self, runbook_doc: Document) -> Dict[str, Any]:
        """Parse runbook document into structured format"""
        
        content = runbook_doc.content
        
        # Simple parser - in production, use structured storage
        sections = {
            "title": runbook_doc.metadata.get("title", "Unknown"),
            "diagnosis_steps": [],
            "remediation_steps": [],
            "prevention": []
        }
        
        current_section = None
        
        for line in content.split('\n'):
            line = line.strip()
            
            if line.startswith("Diagnosis:"):
                current_section = "diagnosis_steps"
            elif line.startswith("Remediation:"):
                current_section = "remediation_steps"
            elif line.startswith("Prevention:"):
                current_section = "prevention"
            elif line.startswith("- ") and current_section:
                sections[current_section].append(line[2:])
        
        return sections
    
    async def upsert(self, documents: List[Document]) -> None:
        """Upsert documents"""
        await self.vector_store.upsert(documents)
    
    async def delete(self, document_ids: List[str]) -> None:
        """Delete documents"""
        await self.vector_store.delete(document_ids)
    
    async def health_check(self) -> bool:
        """Check if retriever is healthy"""
        return await self.vector_store.health_check()