"""
Content Retrieval Agent

Retrieves relevant educational content from curriculum database
"""

from typing import Dict, Any, List, Optional
import logging
from datetime import datetime
import sys
sys.path.append('../..')

from shared.base_agent import BaseAgent, AgentResponse
from rag.curriculum_retriever import CurriculumRetriever

logger = logging.getLogger(__name__)


class ContentRetrievalAgent(BaseAgent):
    """
    Agent for retrieving educational content
    """
    
    def __init__(self):
        super().__init__(
            name="Content Retrieval Agent",
            model="claude-sonnet-4-20250514",
            temperature=0.3
        )
        
        self.retriever = CurriculumRetriever()
    
    def _get_system_prompt(self) -> str:
        """System prompt for content retrieval"""
        return """You are an expert at finding and organizing educational content.

Your role:
1. Retrieve relevant curriculum content for topics
2. Ensure content matches student level
3. Organize content in logical learning sequence
4. Identify prerequisite concepts
5. Find supplementary materials and examples
6. Curate appropriate difficulty level

Output format:
- Main content with clear structure
- Prerequisites identified
- Related concepts
- Practice materials
- Additional resources"""
    
    async def process(self, input_data: Dict[str, Any]) -> AgentResponse:
        """
        Process content retrieval request
        
        Args:
            input_data: {
                "topic": str,
                "grade_level": str,
                "subject": str,
                "content_types": List[str] (optional),
                "difficulty": str (optional)
            }
        
        Returns:
            AgentResponse with retrieved content
        """
        
        topic = input_data.get("topic")
        grade_level = input_data.get("grade_level")
        subject = input_data.get("subject")
        content_types = input_data.get("content_types", ["lesson", "example", "exercise"])
        difficulty = input_data.get("difficulty", "medium")
        
        if not topic:
            raise ValueError("topic is required")
        
        logger.info(f"Retrieving content for {topic} ({grade_level}, {subject})")
        
        # Initialize retriever
        await self.retriever.initialize()
        
        # Search for content
        results = await self.retriever.search(
            topic=topic,
            grade_level=grade_level,
            subject=subject,
            content_types=content_types,
            difficulty=difficulty,
            top_k=10
        )
        
        # Organize and synthesize content
        organized_content = await self._organize_content(
            topic=topic,
            results=results
        )
        
        return AgentResponse(
            content=organized_content,
            agent_name=self.name,
            timestamp=datetime.utcnow(),
            metadata={
                "topic": topic,
                "grade_level": grade_level,
                "subject": subject,
                "content_count": len(results),
                "content_types": content_types
            },
            confidence=0.85,
            sources=[r.document.id for r in results]
        )
    
    async def _organize_content(
        self,
        topic: str,
        results: List[Any]
    ) -> str:
        """Organize retrieved content into coherent lesson"""
        
        if not results:
            return f"No content found for {topic}. Please check the topic name or try a related concept."
        
        # Group content by type
        from collections import defaultdict
        by_type = defaultdict(list)
        
        for result in results:
            content_type = result.document.metadata.get("content_type", "general")
            by_type[content_type].append(result)
        
        # Build organized content
        organized_parts = [f"Learning Content: {topic}\n"]
        
        # Prerequisites
        if "prerequisite" in by_type:
            organized_parts.append("Prerequisites:")
            for result in by_type["prerequisite"][:3]:
                organized_parts.append(f"  • {result.document.metadata.get('title', 'Concept')}")
            organized_parts.append("")
        
        # Main lesson
        if "lesson" in by_type:
            organized_parts.append("Main Content:")
            for result in by_type["lesson"][:2]:
                organized_parts.append(result.document.content[:500] + "...")
            organized_parts.append("")
        
        # Examples
        if "example" in by_type:
            organized_parts.append("Examples:")
            for i, result in enumerate(by_type["example"][:3], 1):
                organized_parts.append(f"{i}. {result.document.content[:200]}...")
            organized_parts.append("")
        
        # Exercises
        if "exercise" in by_type:
            organized_parts.append(f"Practice Exercises ({len(by_type['exercise'])} available)")
            organized_parts.append("")
        
        return "\n".join(organized_parts)