"""
Explanation Agent

Provides clear, multi-modal explanations of concepts
"""

from typing import Dict, Any, List, Optional
import logging
from datetime import datetime
import sys
sys.path.append('../..')

from shared.base_agent import BaseAgent, AgentResponse

logger = logging.getLogger(__name__)


class ExplanationAgent(BaseAgent):
    """
    Agent specialized in generating clear explanations
    """
    
    def __init__(self):
        super().__init__(
            name="Explanation Agent",
            model="claude-sonnet-4-20250514",
            temperature=0.7
        )
    
    def _get_system_prompt(self) -> str:
        """System prompt for explanations"""
        return """You are an expert at explaining complex concepts clearly and simply.

Your role:
1. Break down complex ideas into simple components
2. Use multiple explanation approaches (verbal, visual, analogical)
3. Provide concrete examples before abstract concepts
4. Build from familiar to unfamiliar
5. Check for understanding at each step
6. Adapt explanation style to audience level

Explanation techniques:
- Analogies: Compare to familiar concepts
- Examples: Concrete instances
- Visual descriptions: Diagrams, charts (described in text)
- Step-by-step: Break into sequential steps
- Contrast: Show what it's NOT
- Application: Show real-world uses

Levels of explanation:
- ELI5 (Explain Like I'm 5): Very simple, concrete
- Elementary: Basic concepts, simple language
- Middle School: Some abstraction, more detail
- High School: Abstract thinking, formal terminology
- College: Advanced concepts, precise terminology

Output format:
- Start with the "big picture"
- Break into clear sections
- Use examples throughout
- Describe visual aids when helpful
- End with summary and connections"""
    
    async def process(self, input_data: Dict[str, Any]) -> AgentResponse:
        """
        Process explanation request
        
        Args:
            input_data: {
                "concept": str,
                "level": str (elementary, middle, high_school, college),
                "learning_style": str (visual, auditory, kinesthetic),
                "prior_knowledge": List[str] (optional),
                "specific_question": str (optional)
            }
        
        Returns:
            AgentResponse with explanation
        """
        
        concept = input_data.get("concept")
        level = input_data.get("level", "high_school")
        learning_style = input_data.get("learning_style", "balanced")
        prior_knowledge = input_data.get("prior_knowledge", [])
        specific_question = input_data.get("specific_question")
        
        if not concept:
            raise ValueError("concept is required")
        
        logger.info(f"Generating {level} level explanation for: {concept}")
        
        # Build explanation prompt
        prompt = self._build_explanation_prompt(
            concept=concept,
            level=level,
            learning_style=learning_style,
            prior_knowledge=prior_knowledge,
            specific_question=specific_question
        )
        
        # Generate explanation
        messages = [{"role": "user", "content": prompt}]
        explanation = await self._call_llm(messages)
        
        # Extract explanation components
        components = self._extract_components(explanation)
        
        return AgentResponse(
            content=explanation,
            agent_name=self.name,
            timestamp=datetime.utcnow(),
            metadata={
                "concept": concept,
                "level": level,
                "learning_style": learning_style,
                "components": components
            },
            confidence=0.9
        )
    
    def _build_explanation_prompt(
        self,
        concept: str,
        level: str,
        learning_style: str,
        prior_knowledge: List[str],
        specific_question: Optional[str]
    ) -> str:
        """Build explanation prompt"""
        
        prompt_parts = [
            f"Explain the concept: {concept}",
            f"Level: {level}",
            f"Learning style preference: {learning_style}",
        ]
        
        if prior_knowledge:
            prompt_parts.append(f"\nStudent already knows: {', '.join(prior_knowledge)}")
        
        if specific_question:
            prompt_parts.append(f"\nSpecific question: {specific_question}")
        
        # Add learning style specific guidance
        style_guidance = {
            "visual": "Use visual descriptions, diagrams (described in text), spatial relationships, and color-coding in examples.",
            "auditory": "Use step-by-step verbal explanations, mnemonics, word patterns, and verbal analogies.",
            "kinesthetic": "Use hands-on examples, physical analogies, action-oriented language, and interactive elements."
        }
        
        if learning_style in style_guidance:
            prompt_parts.append(f"\n{style_guidance[learning_style]}")
        
        prompt_parts.append("""
Provide a clear, comprehensive explanation that:
1. Starts with the big picture / core idea
2. Breaks down into understandable components
3. Uses relevant examples
4. Describes helpful visuals (diagrams, charts)
5. Connects to prior knowledge
6. Includes a summary
7. Suggests next concepts to learn""")
        
        return "\n".join(prompt_parts)
    
    def _extract_components(self, explanation: str) -> Dict[str, bool]:
        """Extract explanation components"""
        
        return {
            "has_big_picture": any(word in explanation.lower() for word in ["big picture", "overall", "in summary"]),
            "has_examples": "example" in explanation.lower() or "for instance" in explanation.lower(),
            "has_analogies": "like" in explanation.lower() or "similar to" in explanation.lower(),
            "has_visuals": "diagram" in explanation.lower() or "chart" in explanation.lower() or "visualize" in explanation.lower(),
            "has_steps": any(word in explanation.lower() for word in ["step 1", "first", "second", "then"]),
            "has_summary": "summary" in explanation.lower() or "in conclusion" in explanation.lower(),
            "has_connections": "next" in explanation.lower() or "builds on" in explanation.lower() or "related to" in explanation.lower()
        }
    
    async def generate_analogy(
        self,
        concept: str,
        familiar_domain: str
    ) -> str:
        """
        Generate an analogy for a concept
        
        Args:
            concept: Concept to explain
            familiar_domain: Familiar domain for analogy (sports, cooking, etc.)
        
        Returns:
            Analogy explanation
        """
        
        prompt = f"""Create a clear analogy to explain "{concept}" using the domain of {familiar_domain}.

The analogy should:
1. Map key aspects of {concept} to familiar elements in {familiar_domain}
2. Be accurate and not misleading
3. Help build intuition
4. Be memorable

Format:
- Present the analogy
- Explain the mapping
- Note any limitations of the analogy"""
        
        messages = [{"role": "user", "content": prompt}]
        analogy = await self._call_llm(messages)
        
        return analogy