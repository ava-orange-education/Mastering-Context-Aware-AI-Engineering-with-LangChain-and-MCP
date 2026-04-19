"""
Pedagogical Quality Evaluator

Evaluates the quality of teaching and instructional content
"""

from typing import Dict, Any, List, Optional
import logging
import sys
sys.path.append('../..')

from shared.base_agent import BaseAgent

logger = logging.getLogger(__name__)


class PedagogicalQualityEvaluator:
    """
    Evaluates pedagogical quality of teaching interactions
    """
    
    def __init__(self):
        self.llm = BaseAgent(
            name="Pedagogical Evaluator",
            model="claude-sonnet-4-20250514",
            temperature=0.0
        )
    
    async def evaluate_explanation(
        self,
        explanation: str,
        concept: str,
        student_level: str,
        learning_objectives: List[str]
    ) -> Dict[str, Any]:
        """
        Evaluate quality of an explanation
        
        Args:
            explanation: The explanation text
            concept: Concept being explained
            student_level: Student's grade/skill level
            learning_objectives: Learning objectives
        
        Returns:
            Quality evaluation
        """
        
        prompt = f"""Evaluate this educational explanation for pedagogical quality:

Concept: {concept}
Student Level: {student_level}
Learning Objectives: {', '.join(learning_objectives)}

Explanation:
{explanation}

Evaluate on these criteria (score 0-10 for each):
1. Clarity: Is it clear and easy to understand?
2. Accuracy: Is the information correct?
3. Appropriateness: Matches student level?
4. Structure: Well-organized and logical?
5. Examples: Uses relevant examples?
6. Engagement: Likely to engage students?
7. Completeness: Covers the concept adequately?

Respond in JSON format:
{{
  "clarity": score,
  "accuracy": score,
  "appropriateness": score,
  "structure": score,
  "examples": score,
  "engagement": score,
  "completeness": score,
  "overall_score": average,
  "strengths": ["list of strengths"],
  "improvements": ["list of improvements"],
  "recommendation": "accept|revise|reject"
}}"""
        
        messages = [{"role": "user", "content": prompt}]
        response = await self.llm._call_llm(messages)
        
        # Parse response
        import json
        try:
            cleaned = response.strip()
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:]
            if cleaned.startswith("```"):
                cleaned = cleaned[3:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            
            evaluation = json.loads(cleaned.strip())
        except json.JSONDecodeError:
            evaluation = {
                "overall_score": 5.0,
                "recommendation": "revise",
                "error": "Failed to parse evaluation"
            }
        
        return evaluation
    
    async def evaluate_practice_problem(
        self,
        problem: Dict[str, Any],
        topic: str,
        difficulty: str,
        learning_objectives: List[str]
    ) -> Dict[str, Any]:
        """
        Evaluate quality of a practice problem
        
        Args:
            problem: Problem with question, solution, hints
            topic: Topic area
            difficulty: Target difficulty
            learning_objectives: Learning objectives
        
        Returns:
            Quality evaluation
        """
        
        prompt = f"""Evaluate this practice problem for pedagogical quality:

Topic: {topic}
Target Difficulty: {difficulty}
Learning Objectives: {', '.join(learning_objectives)}

Question: {problem.get('question', '')}
Solution: {problem.get('solution', '')}

Evaluate on:
1. Alignment with objectives (0-10)
2. Difficulty appropriateness (0-10)
3. Clarity of question (0-10)
4. Solution quality (0-10)
5. Educational value (0-10)

Respond in JSON format with scores and brief feedback."""
        
        messages = [{"role": "user", "content": prompt}]
        response = await self.llm._call_llm(messages)
        
        # Parse response
        import json
        try:
            cleaned = response.strip()
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:]
            if cleaned.startswith("```"):
                cleaned = cleaned[3:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            
            evaluation = json.loads(cleaned.strip())
        except:
            evaluation = {"overall_score": 5.0}
        
        return evaluation
    
    def evaluate_feedback_quality(
        self,
        feedback: str,
        student_answer: str,
        correctness: bool
    ) -> Dict[str, Any]:
        """
        Evaluate quality of feedback given to student
        
        Args:
            feedback: Feedback text
            student_answer: Student's answer
            correctness: Whether answer was correct
        
        Returns:
            Feedback quality metrics
        """
        
        quality_metrics = {
            "is_specific": self._check_specificity(feedback),
            "is_constructive": self._check_constructiveness(feedback),
            "has_actionable_steps": self._check_actionable(feedback),
            "is_encouraging": self._check_encouraging(feedback),
            "addresses_misconceptions": self._check_misconceptions(feedback, student_answer, correctness)
        }
        
        # Calculate overall quality
        quality_score = sum(quality_metrics.values()) / len(quality_metrics)
        
        return {
            "quality_score": round(quality_score, 3),
            "metrics": quality_metrics,
            "passes_quality": quality_score >= 0.6
        }
    
    def _check_specificity(self, feedback: str) -> float:
        """Check if feedback is specific (not generic)"""
        
        generic_phrases = ["good job", "try again", "keep going", "nice work"]
        
        # Longer feedback is usually more specific
        if len(feedback) < 50:
            return 0.3
        
        # Check for generic phrases
        has_generic = any(phrase in feedback.lower() for phrase in generic_phrases)
        
        if has_generic and len(feedback) < 100:
            return 0.4
        
        return 0.8
    
    def _check_constructiveness(self, feedback: str) -> float:
        """Check if feedback is constructive"""
        
        constructive_indicators = ["consider", "try", "instead", "you could", "next time"]
        
        has_indicators = any(ind in feedback.lower() for ind in constructive_indicators)
        
        return 0.8 if has_indicators else 0.4
    
    def _check_actionable(self, feedback: str) -> float:
        """Check if feedback has actionable steps"""
        
        action_indicators = ["step", "practice", "review", "focus on", "work on"]
        
        has_actions = any(ind in feedback.lower() for ind in action_indicators)
        
        return 0.8 if has_actions else 0.3
    
    def _check_encouraging(self, feedback: str) -> float:
        """Check if feedback is encouraging"""
        
        encouraging_words = ["progress", "improvement", "well done", "great", "keep", "continue"]
        
        has_encouragement = any(word in feedback.lower() for word in encouraging_words)
        
        return 0.8 if has_encouragement else 0.5
    
    def _check_misconceptions(
        self,
        feedback: str,
        student_answer: str,
        correctness: bool
    ) -> float:
        """Check if feedback addresses misconceptions"""
        
        if correctness:
            # For correct answers, not critical
            return 0.7
        
        # For incorrect answers, should address the error
        addresses_error = any(word in feedback.lower() for word in ["error", "mistake", "misconception", "incorrect", "issue"])
        
        return 0.8 if addresses_error else 0.3