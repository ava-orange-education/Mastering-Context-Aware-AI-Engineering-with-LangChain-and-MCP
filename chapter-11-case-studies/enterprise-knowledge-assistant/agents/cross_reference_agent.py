"""
Cross Reference Agent

Finds related documents and connections across the organization
"""

from typing import Dict, Any, List, Optional
import logging
from datetime import datetime
import sys
sys.path.append('../..')

from shared.base_agent import BaseAgent, AgentResponse
from rag.multi_source_retriever import MultiSourceRetriever

logger = logging.getLogger(__name__)


class CrossReferenceAgent(BaseAgent):
    """
    Agent for finding related documents and cross-references
    """
    
    def __init__(self):
        super().__init__(
            name="Cross Reference Agent",
            model="claude-sonnet-4-20250514",
            temperature=0.3
        )
        
        self.retriever = MultiSourceRetriever()
    
    def _get_system_prompt(self) -> str:
        """System prompt for cross-referencing"""
        return """You are an expert at finding connections between documents and information.

Your role:
1. Identify related documents across different sources
2. Find connections between projects, teams, and initiatives
3. Discover dependencies and prerequisites
4. Map information flow across the organization
5. Highlight contradictions or inconsistencies

Guidelines:
- Look for topical relationships (similar content)
- Identify temporal relationships (related timeframes)
- Find organizational relationships (same team/project)
- Note document relationships (references, updates, versions)
- Discover people relationships (same authors/contributors)
- Flag conflicting information across sources

Output format:
- Related documents with relationship type
- Connection strength (strong, medium, weak)
- Relationship explanation
- Potential conflicts or inconsistencies
- Suggested documents to review next"""
    
    async def process(self, input_data: Dict[str, Any]) -> AgentResponse:
        """
        Process cross-reference request
        
        Args:
            input_data: {
                "document_id": str or "query": str,
                "user_id": str,
                "relationship_types": List[str] (optional),
                "max_results": int (optional),
                "include_sources": List[str] (optional)
            }
        
        Returns:
            AgentResponse with related documents
        """
        
        document_id = input_data.get("document_id")
        query = input_data.get("query")
        user_id = input_data.get("user_id")
        relationship_types = input_data.get("relationship_types", ["all"])
        max_results = input_data.get("max_results", 10)
        include_sources = input_data.get("include_sources", ["all"])
        
        if not document_id and not query:
            raise ValueError("Either document_id or query is required")
        
        logger.info(f"Finding cross-references for user {user_id}")
        
        # Initialize retriever
        await self.retriever.initialize()
        
        # If document_id provided, get document and use as query
        if document_id:
            # In production, fetch document from vector store
            # For now, use document_id as query
            search_query = document_id
        else:
            search_query = query
        
        # Find related documents
        related_results = await self.retriever.search_related(
            query=search_query,
            user_id=user_id,
            top_k=max_results,
            sources=include_sources
        )
        
        # Analyze relationships
        relationships = await self._analyze_relationships(
            primary_query=search_query,
            related_docs=related_results,
            relationship_types=relationship_types
        )
        
        # Generate explanation
        explanation = await self._generate_explanation(
            query=search_query,
            relationships=relationships
        )
        
        return AgentResponse(
            content=explanation,
            agent_name=self.name,
            timestamp=datetime.utcnow(),
            metadata={
                "primary_document": document_id,
                "query": query,
                "total_related": len(related_results),
                "relationships": relationships,
                "relationship_types": relationship_types
            },
            confidence=0.85,
            sources=[r["document_id"] for r in relationships]
        )
    
    async def _analyze_relationships(
        self,
        primary_query: str,
        related_docs: List[Any],
        relationship_types: List[str]
    ) -> List[Dict[str, Any]]:
        """Analyze relationships between documents"""
        
        relationships = []
        
        for result in related_docs:
            doc = result.document
            
            # Determine relationship type
            rel_type = self._determine_relationship_type(
                primary_query,
                doc,
                relationship_types
            )
            
            if rel_type:
                relationships.append({
                    "document_id": doc.id,
                    "title": doc.metadata.get("title", "Untitled"),
                    "source": doc.metadata.get("source", "Unknown"),
                    "relationship_type": rel_type,
                    "relationship_strength": self._calculate_strength(result.score),
                    "relevance_score": round(result.score, 3),
                    "explanation": self._explain_relationship(rel_type, doc)
                })
        
        return relationships
    
    def _determine_relationship_type(
        self,
        primary_query: str,
        doc: Any,
        allowed_types: List[str]
    ) -> Optional[str]:
        """Determine type of relationship"""
        
        # Check metadata for explicit relationships
        doc_type = doc.metadata.get("document_type", "")
        doc_tags = doc.metadata.get("tags", [])
        
        # Topical relationship
        if "all" in allowed_types or "topical" in allowed_types:
            # Documents on similar topics
            return "topical"
        
        # Temporal relationship
        if "temporal" in allowed_types:
            # Documents from similar time period
            if "created_date" in doc.metadata:
                return "temporal"
        
        # Organizational relationship
        if "organizational" in allowed_types:
            # Same team, project, or department
            if "department" in doc.metadata or "team" in doc.metadata:
                return "organizational"
        
        # Default to topical
        return "topical"
    
    def _calculate_strength(self, relevance_score: float) -> str:
        """Calculate relationship strength"""
        
        if relevance_score >= 0.85:
            return "strong"
        elif relevance_score >= 0.7:
            return "medium"
        else:
            return "weak"
    
    def _explain_relationship(
        self,
        relationship_type: str,
        doc: Any
    ) -> str:
        """Explain the relationship"""
        
        explanations = {
            "topical": f"Discusses similar topics: {', '.join(doc.metadata.get('topics', ['related content']))}",
            "temporal": f"Created around the same time ({doc.metadata.get('created_date', 'unknown date')})",
            "organizational": f"From same {doc.metadata.get('department', 'organization')}",
            "sequential": "Part of the same series or workflow",
            "dependency": "Referenced by or depends on this document"
        }
        
        return explanations.get(relationship_type, "Related document")
    
    async def _generate_explanation(
        self,
        query: str,
        relationships: List[Dict[str, Any]]
    ) -> str:
        """Generate human-readable explanation of relationships"""
        
        if not relationships:
            return "No related documents found."
        
        # Group by relationship type
        by_type = {}
        for rel in relationships:
            rel_type = rel["relationship_type"]
            if rel_type not in by_type:
                by_type[rel_type] = []
            by_type[rel_type].append(rel)
        
        # Format explanation
        explanation_parts = [f"Found {len(relationships)} related documents:\n"]
        
        for rel_type, docs in by_type.items():
            explanation_parts.append(f"\n{rel_type.capitalize()} relationships ({len(docs)}):")
            
            for doc in docs[:3]:  # Top 3 per type
                explanation_parts.append(
                    f"  - {doc['title']} ({doc['relationship_strength']} connection)"
                )
                explanation_parts.append(f"    {doc['explanation']}")
        
        return "\n".join(explanation_parts)