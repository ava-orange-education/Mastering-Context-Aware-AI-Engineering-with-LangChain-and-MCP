"""
Incident History Retriever

Retrieves historical incident data for learning and pattern matching
"""

from typing import List, Dict, Any, Optional
import logging
import sys
sys.path.append('../..')

from shared.base_rag import BaseRAG, Document, SearchResult
from shared.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class IncidentHistoryRetriever(BaseRAG):
    """
    Retrieves historical incident data
    """
    
    def __init__(self):
        super().__init__(
            collection_name="incident_history",
            embedding_model="voyage-2"
        )
        
        self.vector_store = None
    
    async def initialize(self) -> None:
        """Initialize vector store connection"""
        
        if settings.vector_store == "pinecone":
            from vector_stores.pinecone_store import PineconeStore
            self.vector_store = PineconeStore()
        else:
            from vector_stores.weaviate_store import WeaviateStore
            self.vector_store = WeaviateStore()
        
        await self.vector_store.initialize()
        
        logger.info("Incident history retriever initialized")
    
    async def search_similar_incidents(
        self,
        incident: Dict[str, Any],
        top_k: int = 5,
        min_similarity: float = 0.7
    ) -> List[SearchResult]:
        """
        Search for similar historical incidents
        
        Args:
            incident: Current incident details
            top_k: Number of results
            min_similarity: Minimum similarity threshold
        
        Returns:
            List of similar incidents
        """
        
        # Build search query
        query = self._build_incident_query(incident)
        
        # Generate embedding
        from ingestion.embedding_pipeline import EmbeddingPipeline
        embedder = EmbeddingPipeline()
        query_embedding = await embedder.generate_embedding(query)
        
        # Build filters
        filters = {}
        
        if "type" in incident:
            filters["incident_type"] = incident["type"]
        
        if "severity" in incident:
            # Include same or higher severity
            severity_order = ["low", "medium", "high", "critical"]
            incident_severity = incident["severity"]
            if incident_severity in severity_order:
                idx = severity_order.index(incident_severity)
                filters["severity"] = {"$in": severity_order[idx:]}
        
        # Search
        results = await self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k * 2,  # Get more, then filter
            filter=filters if filters else None
        )
        
        # Filter by similarity threshold
        filtered_results = [
            r for r in results
            if r.score >= min_similarity
        ]
        
        logger.info(
            f"Found {len(filtered_results)} similar incidents "
            f"(out of {len(results)} total)"
        )
        
        return filtered_results[:top_k]
    
    def _build_incident_query(self, incident: Dict[str, Any]) -> str:
        """Build search query from incident"""
        
        query_parts = []
        
        if "title" in incident:
            query_parts.append(incident["title"])
        
        if "description" in incident:
            query_parts.append(incident["description"])
        
        if "symptoms" in incident:
            if isinstance(incident["symptoms"], list):
                query_parts.append(" ".join(incident["symptoms"]))
            else:
                query_parts.append(str(incident["symptoms"]))
        
        if "affected_components" in incident:
            components = incident["affected_components"]
            if isinstance(components, list):
                query_parts.append("Components: " + ", ".join(components))
        
        if "error_messages" in incident:
            errors = incident["error_messages"]
            if isinstance(errors, list):
                query_parts.extend(errors[:3])  # First 3 errors
        
        return " ".join(query_parts)
    
    async def store_incident(
        self,
        incident: Dict[str, Any],
        resolution: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Store incident in history
        
        Args:
            incident: Incident details
            resolution: Resolution details (if resolved)
        """
        
        # Build document content
        content_parts = [
            f"Incident: {incident.get('title', 'Unknown')}",
            f"Type: {incident.get('type', 'unknown')}",
            f"Severity: {incident.get('severity', 'unknown')}",
        ]
        
        if "description" in incident:
            content_parts.append(f"Description: {incident['description']}")
        
        if "symptoms" in incident:
            symptoms = incident["symptoms"]
            if isinstance(symptoms, list):
                content_parts.append(f"Symptoms: {', '.join(symptoms)}")
        
        if "affected_components" in incident:
            components = incident["affected_components"]
            if isinstance(components, list):
                content_parts.append(f"Affected: {', '.join(components)}")
        
        if resolution:
            if "root_cause" in resolution:
                content_parts.append(f"Root Cause: {resolution['root_cause']}")
            
            if "resolution_steps" in resolution:
                steps = resolution["resolution_steps"]
                if isinstance(steps, list):
                    content_parts.append(f"Resolution: {', '.join(steps)}")
            
            if "actions_taken" in resolution:
                actions = resolution["actions_taken"]
                if isinstance(actions, list):
                    content_parts.append(f"Actions: {', '.join(actions)}")
        
        content = "\n".join(content_parts)
        
        # Generate embedding
        from ingestion.embedding_pipeline import EmbeddingPipeline
        embedder = EmbeddingPipeline()
        embedding = await embedder.generate_embedding(content)
        
        # Create document
        doc = Document(
            id=incident.get("id", f"incident_{datetime.utcnow().timestamp()}"),
            content=content,
            metadata={
                "incident_type": incident.get("type"),
                "severity": incident.get("severity"),
                "affected_components": incident.get("affected_components", []),
                "root_cause": resolution.get("root_cause") if resolution else None,
                "resolution_time": resolution.get("resolution_time") if resolution else None,
                "resolved": resolution is not None,
                "timestamp": incident.get("started_at", datetime.utcnow().isoformat())
            },
            embedding=embedding
        )
        
        # Store
        await self.vector_store.upsert([doc])
        
        logger.info(f"Stored incident in history: {doc.id}")
    
    async def get_incident_by_id(
        self,
        incident_id: str
    ) -> Optional[Document]:
        """Get specific incident by ID"""
        
        results = await self.vector_store.get_by_ids([incident_id])
        
        return results[0] if results else None
    
    async def get_incidents_by_component(
        self,
        component: str,
        top_k: int = 10
    ) -> List[SearchResult]:
        """
        Get incidents affecting a specific component
        
        Args:
            component: Component name
            top_k: Number of results
        
        Returns:
            List of incidents
        """
        
        from ingestion.embedding_pipeline import EmbeddingPipeline
        embedder = EmbeddingPipeline()
        
        query = f"incidents affecting {component}"
        query_embedding = await embedder.generate_embedding(query)
        
        results = await self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k,
            filter={"affected_components": {"$contains": component}}
        )
        
        return results
    
    async def get_recent_incidents(
        self,
        hours: int = 24,
        top_k: int = 50
    ) -> List[Document]:
        """
        Get recent incidents
        
        Args:
            hours: Look back this many hours
            top_k: Maximum results
        
        Returns:
            List of recent incidents
        """
        
        from datetime import datetime, timedelta
        
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        cutoff_iso = cutoff.isoformat()
        
        # In production, would query with timestamp filter
        # For now, return empty list as placeholder
        
        logger.info(f"Querying incidents from last {hours} hours")
        
        return []
    
    async def get_resolution_patterns(
        self,
        incident_type: str
    ) -> Dict[str, Any]:
        """
        Get common resolution patterns for incident type
        
        Args:
            incident_type: Type of incident
        
        Returns:
            Resolution patterns and statistics
        """
        
        from ingestion.embedding_pipeline import EmbeddingPipeline
        embedder = EmbeddingPipeline()
        
        query = f"{incident_type} incident resolutions"
        query_embedding = await embedder.generate_embedding(query)
        
        results = await self.vector_store.search(
            query_embedding=query_embedding,
            top_k=50,
            filter={
                "incident_type": incident_type,
                "resolved": True
            }
        )
        
        # Analyze patterns
        root_causes = {}
        actions_taken = {}
        resolution_times = []
        
        for result in results:
            metadata = result.document.metadata
            
            # Count root causes
            if "root_cause" in metadata and metadata["root_cause"]:
                cause = metadata["root_cause"]
                root_causes[cause] = root_causes.get(cause, 0) + 1
            
            # Count resolution times
            if "resolution_time" in metadata:
                resolution_times.append(metadata["resolution_time"])
        
        # Calculate statistics
        avg_resolution_time = None
        if resolution_times:
            avg_resolution_time = sum(resolution_times) / len(resolution_times)
        
        return {
            "incident_type": incident_type,
            "total_incidents": len(results),
            "common_root_causes": sorted(
                root_causes.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5],
            "avg_resolution_time": avg_resolution_time,
            "resolution_times": resolution_times
        }
    
    async def upsert(self, documents: List[Document]) -> None:
        """Upsert documents"""
        await self.vector_store.upsert(documents)
    
    async def delete(self, document_ids: List[str]) -> None:
        """Delete documents"""
        await self.vector_store.delete(document_ids)
    
    async def health_check(self) -> bool:
        """Check if retriever is healthy"""
        return await self.vector_store.health_check()