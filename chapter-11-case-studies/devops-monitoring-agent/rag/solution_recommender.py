"""
Solution Recommender

Recommends solutions based on historical incident data
"""

from typing import List, Dict, Any, Optional
import logging
from collections import Counter

logger = logging.getLogger(__name__)


class SolutionRecommender:
    """
    Recommends solutions based on incident patterns
    """
    
    def __init__(self):
        self.incident_retriever = None
        self.runbook_retriever = None
    
    async def initialize(self) -> None:
        """Initialize retrievers"""
        
        from .incident_knowledge_retriever import IncidentKnowledgeRetriever
        from .runbook_retriever import RunbookRetriever
        
        self.incident_retriever = IncidentKnowledgeRetriever()
        self.runbook_retriever = RunbookRetriever()
        
        await self.incident_retriever.initialize()
        await self.runbook_retriever.initialize()
        
        logger.info("Solution recommender initialized")
    
    async def recommend_solutions(
        self,
        incident: Dict[str, Any],
        max_solutions: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Recommend solutions for an incident
        
        Args:
            incident: Current incident details
            max_solutions: Maximum solutions to return
        
        Returns:
            List of recommended solutions
        """
        
        solutions = []
        
        # 1. Find similar historical incidents
        similar_incidents = await self.incident_retriever.search_similar_incidents(
            incident=incident,
            top_k=10
        )
        
        # Extract solutions from similar incidents
        for result in similar_incidents:
            if "resolution" in result.document.metadata:
                solutions.append({
                    "solution": result.document.metadata["resolution"],
                    "source": "historical_incident",
                    "source_id": result.document.id,
                    "confidence": result.score,
                    "incident_title": result.document.metadata.get("title"),
                    "resolution_time": result.document.metadata.get("resolution_time")
                })
        
        # 2. Find relevant runbooks
        query = incident.get("title", "") + " " + incident.get("description", "")
        runbooks = await self.runbook_retriever.search_runbooks(
            query=query,
            incident_type=incident.get("type"),
            component=incident.get("affected_components", [None])[0] if incident.get("affected_components") else None,
            top_k=3
        )
        
        # Extract solutions from runbooks
        for result in runbooks:
            runbook = self.runbook_retriever.parse_runbook(result.document)
            
            if runbook.get("remediation_steps"):
                solutions.append({
                    "solution": " → ".join(runbook["remediation_steps"]),
                    "source": "runbook",
                    "source_id": result.document.id,
                    "confidence": result.score,
                    "runbook_title": runbook["title"],
                    "steps": runbook["remediation_steps"]
                })
        
        # 3. Rank and deduplicate solutions
        ranked_solutions = self._rank_solutions(solutions)
        
        return ranked_solutions[:max_solutions]
    
    def _rank_solutions(
        self,
        solutions: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Rank solutions by effectiveness and confidence"""
        
        # Score each solution
        for solution in solutions:
            score = 0.0
            
            # Confidence score
            score += solution.get("confidence", 0.5) * 0.4
            
            # Source preference (historical incidents > runbooks)
            if solution["source"] == "historical_incident":
                score += 0.3
            elif solution["source"] == "runbook":
                score += 0.2
            
            # Resolution time (faster is better)
            if "resolution_time" in solution:
                # Normalize resolution time (assuming minutes)
                res_time = solution["resolution_time"]
                if res_time < 5:
                    score += 0.3
                elif res_time < 15:
                    score += 0.2
                elif res_time < 60:
                    score += 0.1
            
            solution["rank_score"] = score
        
        # Sort by score
        solutions.sort(key=lambda x: x["rank_score"], reverse=True)
        
        return solutions
    
    async def get_common_resolutions(
        self,
        incident_type: str,
        component: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get most common resolutions for incident type
        
        Args:
            incident_type: Type of incident
            component: Optional component filter
        
        Returns:
            List of common resolutions with frequency
        """
        
        # Search for incidents of this type
        from ingestion.embedding_pipeline import EmbeddingPipeline
        embedder = EmbeddingPipeline()
        
        query = f"{incident_type} incidents"
        if component:
            query += f" affecting {component}"
        
        query_embedding = await embedder.generate_embedding(query)
        
        results = await self.incident_retriever.vector_store.search(
            query_embedding=query_embedding,
            top_k=50,
            filter={"incident_type": incident_type}
        )
        
        # Count resolutions
        resolutions = []
        for result in results:
            if "resolution" in result.document.metadata:
                resolutions.append(result.document.metadata["resolution"])
        
        # Find most common
        resolution_counts = Counter(resolutions)
        
        common_resolutions = [
            {
                "resolution": resolution,
                "frequency": count,
                "percentage": count / len(resolutions) if resolutions else 0
            }
            for resolution, count in resolution_counts.most_common(10)
        ]
        
        return common_resolutions
    
    async def analyze_resolution_effectiveness(
        self,
        incident_type: str
    ) -> Dict[str, Any]:
        """
        Analyze effectiveness of different resolutions
        
        Args:
            incident_type: Type of incident
        
        Returns:
            Analysis of resolution effectiveness
        """
        
        # Get incidents of this type
        from ingestion.embedding_pipeline import EmbeddingPipeline
        embedder = EmbeddingPipeline()
        
        query_embedding = await embedder.generate_embedding(f"{incident_type} incidents")
        
        results = await self.incident_retriever.vector_store.search(
            query_embedding=query_embedding,
            top_k=100,
            filter={"incident_type": incident_type}
        )
        
        # Group by resolution
        resolution_metrics = {}
        
        for result in results:
            metadata = result.document.metadata
            
            resolution = metadata.get("resolution")
            if not resolution:
                continue
            
            if resolution not in resolution_metrics:
                resolution_metrics[resolution] = {
                    "count": 0,
                    "resolution_times": [],
                    "recurrence_rate": 0
                }
            
            resolution_metrics[resolution]["count"] += 1
            
            if "resolution_time" in metadata:
                resolution_metrics[resolution]["resolution_times"].append(
                    metadata["resolution_time"]
                )
        
        # Calculate statistics
        for resolution, metrics in resolution_metrics.items():
            if metrics["resolution_times"]:
                import statistics
                metrics["avg_resolution_time"] = statistics.mean(metrics["resolution_times"])
                metrics["median_resolution_time"] = statistics.median(metrics["resolution_times"])
        
        return {
            "incident_type": incident_type,
            "total_incidents": len(results),
            "unique_resolutions": len(resolution_metrics),
            "resolutions": resolution_metrics
        }