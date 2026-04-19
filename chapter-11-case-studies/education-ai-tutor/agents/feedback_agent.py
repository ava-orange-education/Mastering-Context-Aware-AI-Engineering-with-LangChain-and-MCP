"""
Feedback Agent

Provides constructive, personalized feedback on student work
"""

from typing import Dict, Any, List, Optional
import logging
from datetime import datetime
import sys
sys.path.append('../..')

from shared.base_agent import BaseAgent, AgentResponse

logger = logging.getLogger(__name__)


class FeedbackAgent(BaseAgent):
    """
    Agent for generating personalized feedback
    """
    
    def __init__(self):
        super().__init__(
            name="Feedback Agent",
            model="claude-sonnet-4-20250514",
            temperature=0.7
        )
    
    def _get_system_prompt(self) -> str:
        """System prompt for feedback"""
        return """You are an expert educator providing constructive feedback to students.

Your role:
1. Give specific, actionable feedback
2. Balance praise and constructive criticism
3. Focus on growth and improvement
4. Be encouraging and supportive
5. Identify both strengths and areas for growth
6. Provide concrete next steps

Feedback principles:
- Be specific, not generic ("Good job" → "Your step-by-step solution shows clear logical thinking")
- Focus on the work, not the person
- Praise effort and strategy, not just correctness
- Point out what's working before what needs improvement
- Make suggestions, don't just point out errors
- Connect feedback to learning goals
- Be timely and relevant

Feedback structure:
1. Acknowledgment: What the student did well
2. Analysis: What was effective in their approach
3. Growth area: One or two specific things to improve
4. Actionable steps: Concrete suggestions
5. Encouragement: Forward-looking, growth-oriented

Tone:
- Encouraging and supportive
- Respectful and professional
- Clear and specific
- Growth-oriented
- Age-appropriate"""
    
    async def process(self, input_data: Dict[str, Any]) -> AgentResponse:
        """
        Process feedback request
        
        Args:
            input_data: {
                "student_work": Dict,
                "rubric": Dict (optional),
                "learning_goals": List[str] (optional),
                "previous_feedback": str (optional),
                "student_level": str (optional)
            }
        
        Returns:
            AgentResponse with feedback
        """
        
        student_work = input_data.get("student_work")
        rubric = input_data.get("rubric")
        learning_goals = input_data.get("learning_goals", [])
        previous_feedback = input_data.get("previous_feedback")
        student_level = input_data.get("student_level", "high_school")
        
        if not student_work:
            raise ValueError("student_work is required")
        
        logger.info("Generating personalized feedback")
        
        # Generate feedback
        feedback = await self._generate_feedback(
            student_work=student_work,
            rubric=rubric,
            learning_goals=learning_goals,
            previous_feedback=previous_feedback,
            student_level=student_level
        )
        
        # Extract feedback components
        components = self._extract_feedback_components(feedback)
        
        return AgentResponse(
            content=feedback,
            agent_name=self.name,
            timestamp=datetime.utcnow(),
            metadata={
                "components": components,
                "student_level": student_level
            },
            confidence=0.9
        )
    
    async def _generate_feedback(
        self,
        student_work: Dict[str, Any],
        rubric: Optional[Dict[str, Any]],
        learning_goals: List[str],
        previous_feedback: Optional[str],
        student_level: str
    ) -> str:
        """Generate personalized feedback"""
        
        # Build feedback prompt
        prompt_parts = [
            "Provide constructive feedback on this student work:",
            f"\nStudent Level: {student_level}",
        ]
        
        # Add student work
        if isinstance(student_work, dict):
            if "question" in student_work:
                prompt_parts.append(f"\nQuestion: {student_work['question']}")
            if "answer" in student_work:
                prompt_parts.append(f"\nStudent's Answer: {student_work['answer']}")
            if "work_shown" in student_work:
                prompt_parts.append(f"\nWork Shown: {student_work['work_shown']}")
        else:
            prompt_parts.append(f"\nStudent Work: {student_work}")
        
        # Add rubric if provided
        if rubric:
            prompt_parts.append(f"\nRubric: {rubric}")
        
        # Add learning goals
        if learning_goals:
            prompt_parts.append(f"\nLearning Goals: {', '.join(learning_goals)}")
        
        # Add previous feedback context
        if previous_feedback:
            prompt_parts.append(f"\nPrevious Feedback Given: {previous_feedback}")
            prompt_parts.append("Note: Acknowledge any improvement since previous feedback.")
        
        prompt_parts.append("""
Provide feedback that:
1. Acknowledges specific strengths
2. Identifies one or two key areas for improvement
3. Gives concrete, actionable suggestions
4. Encourages continued effort
5. Connects to learning goals

Use a warm, encouraging tone appropriate for the student's level.""")
        
        prompt = "\n".join(prompt_parts)
        
        messages = [{"role": "user", "content": prompt}]
        feedback = await self._call_llm(messages)
        
        return feedback
    
    def _extract_feedback_components(self, feedback: str) -> Dict[str, bool]:
        """Extract feedback components"""
        
        feedback_lower = feedback.lower()
        
        return {
            "has_acknowledgment": any(word in feedback_lower for word in ["well done", "good", "excellent", "great"]),
            "has_specific_praise": any(word in feedback_lower for word in ["specific", "particularly", "especially"]),
            "has_growth_area": any(word in feedback_lower for word in ["improve", "develop", "work on", "consider"]),
            "has_actionable_steps": any(word in feedback_lower for word in ["try", "practice", "next time", "you could"]),
            "has_encouragement": any(word in feedback_lower for word in ["keep", "continue", "progress", "growth"]),
            "is_specific": len(feedback) > 200  # Detailed feedback
        }
    
    async def generate_hint(
        self,
        problem: str,
        student_attempt: str,
        hint_level: str = "light"
    ) -> str:
        """
        Generate a hint for a student
        
        Args:
            problem: The problem
            student_attempt: Student's attempt so far
            hint_level: light, medium, or strong
        
        Returns:
            Hint text
        """
        
        hint_instructions = {
            "light": "Give a very subtle hint that points them in the right direction without giving away the answer. Ask a guiding question.",
            "medium": "Provide a hint that narrows down the approach. Suggest a strategy or show the first step.",
            "strong": "Give a detailed hint that shows most of the solution process but leaves the final step for the student."
        }
        
        prompt = f"""Problem: {problem}

Student's Attempt: {student_attempt}

{hint_instructions.get(hint_level, hint_instructions['medium'])}

The hint should:
- Help the student discover the answer themselves
- Not simply give the answer
- Build their confidence
- Teach the underlying concept"""
        
        messages = [{"role": "user", "content": prompt}]
        hint = await self._call_llm(messages)
        
        return hint
    
    async def generate_encouragement(
        self,
        context: str,
        student_emotion: str = "frustrated"
    ) -> str:
        """
        Generate encouraging message
        
        Args:
            context: Current situation
            student_emotion: Student's emotional state
        
        Returns:
            Encouraging message
        """
        
        prompt = f"""The student is feeling {student_emotion} in this situation:
{context}

Provide a brief, encouraging message that:
1. Validates their feelings
2. Reminds them that struggle is part of learning
3. Encourages them to keep trying
4. Suggests a positive next step

Keep it warm, authentic, and brief (2-3 sentences)."""
        
        messages = [{"role": "user", "content": prompt}]
        encouragement = await self._call_llm(messages)
        
        return encouragement