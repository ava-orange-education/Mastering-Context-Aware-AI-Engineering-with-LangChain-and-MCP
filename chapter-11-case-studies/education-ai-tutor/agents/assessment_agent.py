"""
Assessment Agent

Evaluates student understanding and provides detailed feedback
"""

from typing import Dict, Any, List, Optional
import logging
from datetime import datetime
import sys
sys.path.append('../..')

from shared.base_agent import BaseAgent, AgentResponse

logger = logging.getLogger(__name__)


class AssessmentAgent(BaseAgent):
    """
    Agent for assessing student knowledge and understanding
    """
    
    def __init__(self):
        super().__init__(
            name="Assessment Agent",
            model="claude-sonnet-4-20250514",
            temperature=0.3
        )
    
    def _get_system_prompt(self) -> str:
        """System prompt for assessment"""
        return """You are an expert educational assessor who evaluates student understanding.

Your role:
1. Evaluate student responses for correctness and understanding
2. Identify specific misconceptions and errors
3. Provide constructive, encouraging feedback
4. Assess depth of understanding, not just correctness
5. Recognize partial understanding and give partial credit
6. Identify learning gaps and prerequisite knowledge issues
7. Recommend next steps for improvement

Assessment principles:
- Look beyond surface-level correctness
- Identify the reasoning behind errors
- Recognize creative or alternative approaches
- Provide specific, actionable feedback
- Be encouraging while being honest
- Focus on growth and improvement
- Distinguish between careless mistakes and conceptual misunderstanding

Output format:
- Overall assessment (correct/partially correct/incorrect)
- Detailed evaluation of student's reasoning
- Specific strengths demonstrated
- Specific areas for improvement
- Misconceptions identified
- Recommended next steps
- Mastery level estimate (0.0 to 1.0)"""
    
    async def process(self, input_data: Dict[str, Any]) -> AgentResponse:
        """
        Process assessment request
        
        Args:
            input_data: {
                "student_id": str,
                "topic": str,
                "questions": List[Dict],
                "student_answers": List[Dict],
                "rubric": Dict (optional)
            }
        
        Returns:
            AgentResponse with assessment results
        """
        
        student_id = input_data.get("student_id")
        topic = input_data.get("topic")
        questions = input_data.get("questions", [])
        student_answers = input_data.get("student_answers", [])
        rubric = input_data.get("rubric")
        
        if not student_id or not topic:
            raise ValueError("student_id and topic are required")
        
        logger.info(f"Assessing {len(student_answers)} responses for topic: {topic}")
        
        # Evaluate each response
        evaluations = []
        
        for i, (question, answer) in enumerate(zip(questions, student_answers)):
            evaluation = await self._evaluate_response(
                question=question,
                student_answer=answer,
                rubric=rubric
            )
            evaluations.append(evaluation)
        
        # Calculate overall metrics
        metrics = self._calculate_metrics(evaluations)
        
        # Generate summary feedback
        summary = self._generate_summary(
            topic=topic,
            evaluations=evaluations,
            metrics=metrics
        )
        
        return AgentResponse(
            content=summary,
            agent_name=self.name,
            timestamp=datetime.utcnow(),
            metadata={
                "student_id": student_id,
                "topic": topic,
                "total_questions": len(questions),
                "evaluations": evaluations,
                "metrics": metrics
            },
            confidence=0.9
        )
    
    async def _evaluate_response(
        self,
        question: Dict[str, Any],
        student_answer: Dict[str, Any],
        rubric: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Evaluate a single student response"""
        
        question_text = question.get("question", "")
        correct_answer = question.get("correct_answer", "")
        student_response = student_answer.get("answer", "")
        
        # Build evaluation prompt
        prompt = f"""Question: {question_text}

Correct Answer: {correct_answer}

Student's Answer: {student_response}

Evaluate this response:
1. Is it correct? (fully, partially, or incorrect)
2. What understanding does the student demonstrate?
3. What misconceptions or errors are present?
4. What specific feedback would help?

Respond in JSON format:
{{
  "correctness": "correct|partial|incorrect",
  "score": 0.0-1.0,
  "strengths": ["list of strengths"],
  "weaknesses": ["list of weaknesses"],
  "misconceptions": ["list of misconceptions"],
  "feedback": "specific feedback for student",
  "next_steps": ["recommended next steps"]
}}"""
        
        if rubric:
            prompt += f"\n\nRubric: {rubric}"
        
        messages = [{"role": "user", "content": prompt}]
        response = await self._call_llm(messages)
        
        # Parse response
        import json
        try:
            # Remove markdown code blocks if present
            cleaned = response.strip()
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:]
            if cleaned.startswith("```"):
                cleaned = cleaned[3:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            
            evaluation = json.loads(cleaned.strip())
        except json.JSONDecodeError:
            # Fallback if JSON parsing fails
            evaluation = {
                "correctness": "partial",
                "score": 0.5,
                "strengths": [],
                "weaknesses": [],
                "misconceptions": [],
                "feedback": response,
                "next_steps": []
            }
        
        return evaluation
    
    def _calculate_metrics(
        self,
        evaluations: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Calculate overall assessment metrics"""
        
        if not evaluations:
            return {
                "total_score": 0.0,
                "average_score": 0.0,
                "mastery_level": 0.0,
                "correct_count": 0,
                "partial_count": 0,
                "incorrect_count": 0
            }
        
        scores = [e.get("score", 0) for e in evaluations]
        
        correct_count = sum(1 for e in evaluations if e.get("correctness") == "correct")
        partial_count = sum(1 for e in evaluations if e.get("correctness") == "partial")
        incorrect_count = sum(1 for e in evaluations if e.get("correctness") == "incorrect")
        
        total_score = sum(scores)
        average_score = total_score / len(scores)
        
        # Calculate mastery level (weighted by consistency)
        mastery_level = self._calculate_mastery_level(scores)
        
        return {
            "total_score": round(total_score, 2),
            "average_score": round(average_score, 3),
            "mastery_level": round(mastery_level, 3),
            "correct_count": correct_count,
            "partial_count": partial_count,
            "incorrect_count": incorrect_count,
            "accuracy_rate": round(correct_count / len(evaluations), 3)
        }
    
    def _calculate_mastery_level(self, scores: List[float]) -> float:
        """
        Calculate mastery level considering consistency
        High variance = lower mastery even with good average
        """
        
        if not scores:
            return 0.0
        
        import statistics
        
        avg = statistics.mean(scores)
        
        # Consider variance - consistent performance indicates mastery
        if len(scores) > 1:
            variance = statistics.variance(scores)
            consistency_factor = max(0, 1 - variance)
        else:
            consistency_factor = 1.0
        
        # Mastery = average performance * consistency
        mastery = avg * (0.7 + 0.3 * consistency_factor)
        
        return min(1.0, mastery)
    
    def _generate_summary(
        self,
        topic: str,
        evaluations: List[Dict[str, Any]],
        metrics: Dict[str, Any]
    ) -> str:
        """Generate summary feedback"""
        
        summary_parts = [
            f"Assessment Summary: {topic}",
            f"\nOverall Performance:",
            f"  Score: {metrics['average_score']:.1%}",
            f"  Mastery Level: {metrics['mastery_level']:.1%}",
            f"  Correct: {metrics['correct_count']}/{len(evaluations)}",
        ]
        
        # Collect common strengths and weaknesses
        all_strengths = []
        all_weaknesses = []
        all_misconceptions = []
        
        for eval in evaluations:
            all_strengths.extend(eval.get("strengths", []))
            all_weaknesses.extend(eval.get("weaknesses", []))
            all_misconceptions.extend(eval.get("misconceptions", []))
        
        # Deduplicate and get top items
        unique_strengths = list(set(all_strengths))[:3]
        unique_weaknesses = list(set(all_weaknesses))[:3]
        unique_misconceptions = list(set(all_misconceptions))[:3]
        
        if unique_strengths:
            summary_parts.append("\nStrengths Demonstrated:")
            for strength in unique_strengths:
                summary_parts.append(f"  ✓ {strength}")
        
        if unique_weaknesses:
            summary_parts.append("\nAreas for Improvement:")
            for weakness in unique_weaknesses:
                summary_parts.append(f"  • {weakness}")
        
        if unique_misconceptions:
            summary_parts.append("\nMisconceptions to Address:")
            for misconception in unique_misconceptions:
                summary_parts.append(f"  ⚠ {misconception}")
        
        # Recommendations
        summary_parts.append("\nRecommended Next Steps:")
        if metrics["mastery_level"] >= 0.8:
            summary_parts.append("  • Ready to advance to more challenging material")
            summary_parts.append("  • Consider exploring advanced applications")
        elif metrics["mastery_level"] >= 0.6:
            summary_parts.append("  • Practice more problems to strengthen understanding")
            summary_parts.append("  • Review specific areas noted above")
        else:
            summary_parts.append("  • Review fundamental concepts")
            summary_parts.append("  • Work with tutor on identified misconceptions")
            summary_parts.append("  • Practice with simpler problems first")
        
        return "\n".join(summary_parts)