"""
Expert Finder Agent

Identifies subject matter experts based on document authorship and contributions
"""

from typing import Dict, Any, List, Optional
import logging
from datetime import datetime
from collections import Counter
import sys
sys.path.append('../..')

from shared.base_agent import BaseAgent, AgentResponse
from rag.multi_source_retriever import MultiSourceRetriever

logger = logging.getLogger(__name__)


class ExpertFinderAgent(BaseAgent):
    """
    Agent for finding subject matter experts
    """
    
    def __init__(self):
        super().__init__(
            name="Expert Finder Agent",
            model="claude-sonnet-4-20250514",
            temperature=0.3
        )
        
        self.retriever = MultiSourceRetriever()
    
    def _get_system_prompt(self) -> str:
        """System prompt for expert finding"""
        return """You are an expert at identifying subject matter experts within an organization.

Your role:
1. Analyze document authorship and contributions
2. Identify knowledge domains and expertise areas
3. Assess depth and breadth of expertise
4. Consider recency and relevance of contributions
5. Recommend appropriate experts for consultation

Guidelines:
- Weight recent contributions higher
- Consider quality and depth of content
- Look for consistent contributions over time
- Note collaboration patterns
- Identify both technical and domain experts
- Distinguish between authors and contributors

Output format:
- Expert name and contact
- Expertise areas
- Contribution summary (count, recency, quality)
- Relevant documents/projects
- Expertise level (beginner, intermediate, expert, thought leader)
- Recommended for (specific use cases)"""
    
    async def process(self, input_data: Dict[str, Any]) -> AgentResponse:
        """
        Process expert finding request
        
        Args:
            input_data: {
                "topic": str,
                "department": str (optional),
                "min_contributions": int (optional),
                "recency_days": int (optional),
                "max_experts": int (optional)
            }
        
        Returns:
            AgentResponse with expert recommendations
        """
        
        topic = input_data.get("topic")
        department = input_data.get("department")
        min_contributions = input_data.get("min_contributions", 3)
        recency_days = input_data.get("recency_days", 365)
        max_experts = input_data.get("max_experts", 5)
        
        if not topic:
            raise ValueError("Topic is required for expert finding")
        
        logger.info(f"Finding experts for topic: {topic}")
        
        # Initialize retriever
        await self.retriever.initialize()
        
        # Search for documents on topic
        search_results = await self.retriever.search_all_sources(
            query=topic,
            user_id="system",  # System search for expert finding
            top_k=50  # Get more results to analyze authorship
        )
        
        # Analyze authorship
        experts = self._analyze_authorship(
            search_results,
            topic,
            department,
            min_contributions,
            recency_days
        )
        
        # Rank experts
        ranked_experts = self._rank_experts(experts)[:max_experts]
        
        # Generate recommendations
        recommendations = await self._generate_recommendations(
            topic=topic,
            experts=ranked_experts
        )
        
        return AgentResponse(
            content=recommendations,
            agent_name=self.name,
            timestamp=datetime.utcnow(),
            metadata={
                "topic": topic,
                "department": department,
                "documents_analyzed": len(search_results),
                "experts_found": len(ranked_experts),
                "experts": ranked_experts
            },
            confidence=self._calculate_confidence(ranked_experts)
        )
    
    def _analyze_authorship(
        self,
        search_results: List[Any],
        topic: str,
        department: Optional[str],
        min_contributions: int,
        recency_days: int
    ) -> List[Dict[str, Any]]:
        """Analyze authorship patterns"""
        
        # Count contributions by author
        author_contributions = {}
        
        for result in search_results:
            doc = result.document
            author = doc.metadata.get("author")
            
            if not author:
                continue
            
            # Filter by department if specified
            doc_department = doc.metadata.get("department")
            if department and doc_department != department:
                continue
            
            if author not in author_contributions:
                author_contributions[author] = {
                    "author": author,
                    "email": doc.metadata.get("author_email"),
                    "department": doc_department,
                    "contributions": [],
                    "total_relevance": 0.0,
                    "recent_contributions": 0
                }
            
            # Add contribution
            contribution = {
                "document_id": doc.id,
                "title": doc.metadata.get("title", "Untitled"),
                "date": doc.metadata.get("created_date"),
                "relevance": result.score
            }
            
            author_contributions[author]["contributions"].append(contribution)
            author_contributions[author]["total_relevance"] += result.score
            
            # Check if recent
            # Simplified - in production, parse and compare dates
            if "2024" in str(contribution["date"]) or "2025" in str(contribution["date"]):
                author_contributions[author]["recent_contributions"] += 1
        
        # Filter by minimum contributions
        experts = [
            expert for expert in author_contributions.values()
            if len(expert["contributions"]) >= min_contributions
        ]
        
        return experts
    
    def _rank_experts(self, experts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Rank experts by expertise indicators"""
        
        for expert in experts:
            # Calculate expertise score
            num_contributions = len(expert["contributions"])
            avg_relevance = expert["total_relevance"] / num_contributions
            recency_ratio = expert["recent_contributions"] / num_contributions
            
            # Weighted score
            expert["expertise_score"] = (
                (num_contributions / 10) * 0.3 +  # Quantity (normalized)
                avg_relevance * 0.4 +              # Quality
                recency_ratio * 0.3                # Recency
            )
            
            # Determine expertise level
            if expert["expertise_score"] >= 0.8:
                expert["expertise_level"] = "thought_leader"
            elif expert["expertise_score"] >= 0.6:
                expert["expertise_level"] = "expert"
            elif expert["expertise_score"] >= 0.4:
                expert["expertise_level"] = "intermediate"
            else:
                expert["expertise_level"] = "contributor"
        
        # Sort by expertise score
        experts.sort(key=lambda x: x["expertise_score"], reverse=True)
        
        return experts
    
    async def _generate_recommendations(
        self,
        topic: str,
        experts: List[Dict[str, Any]]
    ) -> str:
        """Generate expert recommendations"""
        
        if not experts:
            return f"No experts found for topic: {topic}"
        
        recommendation_parts = [
            f"Subject Matter Experts for '{topic}':\n"
        ]
        
        for i, expert in enumerate(experts, 1):
            recommendation_parts.append(
                f"\n{i}. {expert['author']}"
            )
            
            if expert["email"]:
                recommendation_parts.append(f"   Email: {expert['email']}")
            
            if expert["department"]:
                recommendation_parts.append(f"   Department: {expert['department']}")
            
            recommendation_parts.append(
                f"   Expertise Level: {expert['expertise_level'].replace('_', ' ').title()}"
            )
            
            recommendation_parts.append(
                f"   Contributions: {len(expert['contributions'])} documents "
                f"({expert['recent_contributions']} recent)"
            )
            
            recommendation_parts.append(
                f"   Expertise Score: {expert['expertise_score']:.2f}"
            )
            
            # Top contributions
            top_contributions = sorted(
                expert["contributions"],
                key=lambda x: x["relevance"],
                reverse=True
            )[:2]
            
            recommendation_parts.append("   Key Contributions:")
            for contrib in top_contributions:
                recommendation_parts.append(f"     - {contrib['title']}")
        
        return "\n".join(recommendation_parts)
    
    def _calculate_confidence(self, experts: List[Dict[str, Any]]) -> float:
        """Calculate confidence in expert recommendations"""
        
        if not experts:
            return 0.0
        
        # Confidence based on:
        # - Number of experts found
        # - Quality of top expert
        # - Distribution of expertise
        
        num_experts_factor = min(len(experts) / 5, 1.0)
        top_expert_score = experts[0]["expertise_score"] if experts else 0.0
        
        confidence = (num_experts_factor * 0.4) + (top_expert_score * 0.6)
        
        return round(confidence, 2)