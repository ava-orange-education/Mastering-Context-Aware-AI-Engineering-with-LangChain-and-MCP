"""
Teaching Agent

Main agent for providing personalized instruction and explanations
"""

from typing import Dict, Any, List, Optional
import logging
from datetime import datetime
import sys
sys.path.append('../..')

from shared.base_agent import BaseAgent, AgentResponse
from student_model.learning_profile import LearningProfile

logger = logging.getLogger(__name__)


class TeachingAgent(BaseAgent):
    """
    Agent for personalized teaching and instruction
    """
    
    def __init__(self):
        super().__init__(
            name="Teaching Agent",
            model="claude-sonnet-4-20250514",
            temperature=0.7
        )
    
    def _get_system_prompt(self) -> str:
        """System prompt for teaching"""
        return """You are an expert AI tutor who provides personalized education.

Your role:
1. Explain concepts clearly and at the appropriate level
2. Adapt to student's learning style (visual, auditory, kinesthetic)
3. Use examples relevant to student's interests
4. Check for understanding throughout
5. Identify and address misconceptions
6. Build on prior knowledge
7. Encourage critical thinking

Teaching principles:
- Start with what the student knows
- Break complex topics into manageable chunks
- Use concrete examples before abstract concepts
- Provide multiple representations (verbal, visual, mathematical)
- Ask questions to promote active learning
- Celebrate progress and effort
- Be patient and encouraging

Learning styles:
- Visual: Use diagrams, charts, color coding, spatial organization
- Auditory: Use step-by-step verbal explanations, mnemonics, analogies
- Kinesthetic: Use hands-on examples, physical analogies, interactive elements

Output format:
- Clear, structured explanation
- Relevant examples
- Visual aids when appropriate (described in text)
- Check-for-understanding questions
- Connection to real-world applications"""
    
    async def process(self, input_data: Dict[str, Any]) -> AgentResponse:
        """
        Process teaching request
        
        Args:
            input_data: {
                "student_profile": LearningProfile or dict,
                "topic": str,
                "question": str (optional),
                "context": str (optional),
                "previous_attempts": list (optional)
            }
        
        Returns:
            AgentResponse with personalized explanation
        """
        
        # Extract input
        student_profile = input_data.get("student_profile")
        topic = input_data.get("topic")
        question = input_data.get("question")
        context = input_data.get("context", "")
        previous_attempts = input_data.get("previous_attempts", [])
        
        if not topic:
            raise ValueError("Topic is required")
        
        # Parse student profile
        if isinstance(student_profile, dict):
            grade_level = student_profile.get("grade_level", "unknown")
            learning_style = student_profile.get("learning_style", "balanced")
            interests = student_profile.get("interests", [])
            mastery_levels = student_profile.get("mastery_levels", {})
        else:
            grade_level = getattr(student_profile, "grade_level", "unknown")
            learning_style = getattr(student_profile, "learning_style", "balanced")
            interests = getattr(student_profile, "interests", [])
            mastery_levels = getattr(student_profile, "mastery_levels", {})
        
        logger.info(
            f"Teaching {topic} to {grade_level} student "
            f"(learning style: {learning_style})"
        )
        
        # Build teaching prompt
        teaching_prompt = self._build_teaching_prompt(
            topic=topic,
            question=question,
            grade_level=grade_level,
            learning_style=learning_style,
            interests=interests,
            mastery_levels=mastery_levels,
            context=context,
            previous_attempts=previous_attempts
        )
        
        # Generate explanation
        messages = [{"role": "user", "content": teaching_prompt}]
        explanation = await self._call_llm(messages)
        
        # Extract teaching elements
        teaching_elements = self._extract_teaching_elements(explanation)
        
        return AgentResponse(
            content=explanation,
            agent_name=self.name,
            timestamp=datetime.utcnow(),
            metadata={
                "topic": topic,
                "grade_level": grade_level,
                "learning_style": learning_style,
                "teaching_elements": teaching_elements
            },
            confidence=0.85
        )
    
    def _build_teaching_prompt(
        self,
        topic: str,
        question: Optional[str],
        grade_level: str,
        learning_style: str,
        interests: List[str],
        mastery_levels: Dict[str, float],
        context: str,
        previous_attempts: List[Dict[str, Any]]
    ) -> str:
        """Build personalized teaching prompt"""
        
        prompt_parts = [
            f"Topic: {topic}",
            f"Grade Level: {grade_level}",
            f"Learning Style: {learning_style}",
        ]
        
        if question:
            prompt_parts.append(f"\nStudent Question: {question}")
        
        if interests:
            prompt_parts.append(f"\nStudent Interests: {', '.join(interests)}")
        
        if mastery_levels:
            prompt_parts.append("\nCurrent Mastery Levels:")
            for subject, level in mastery_levels.items():
                prompt_parts.append(f"  - {subject}: {level:.0%}")
        
        if context:
            prompt_parts.append(f"\nContext: {context}")
        
        if previous_attempts:
            prompt_parts.append("\nPrevious Attempts:")
            for i, attempt in enumerate(previous_attempts, 1):
                prompt_parts.append(
                    f"  {i}. {attempt.get('answer', 'No answer')} "
                    f"({'Correct' if attempt.get('correct') else 'Incorrect'})"
                )
        
        prompt_parts.append(f"""
Please provide a personalized explanation that:
1. Matches the student's {grade_level} level
2. Adapts to {learning_style} learning style
3. Uses examples from their interests when relevant
4. Builds on their current knowledge
5. Addresses any misconceptions from previous attempts
6. Includes practice examples
7. Asks check-for-understanding questions""")
        
        return "\n".join(prompt_parts)
    
    def _extract_teaching_elements(self, explanation: str) -> Dict[str, Any]:
        """Extract teaching elements from explanation"""
        
        elements = {
            "has_examples": "example:" in explanation.lower() or "for instance" in explanation.lower(),
            "has_visuals": "diagram" in explanation.lower() or "graph" in explanation.lower(),
            "has_questions": "?" in explanation,
            "has_steps": any(word in explanation.lower() for word in ["step 1", "first,", "second,"]),
            "has_real_world": "real world" in explanation.lower() or "in practice" in explanation.lower()
        }
        
        return elements