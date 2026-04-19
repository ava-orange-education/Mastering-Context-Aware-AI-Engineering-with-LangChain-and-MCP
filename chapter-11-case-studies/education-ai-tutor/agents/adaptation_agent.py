"""
Adaptation Agent

Dynamically adapts content difficulty and presentation based on student performance
"""

from typing import Dict, Any, List, Optional
import logging
from datetime import datetime
import sys
sys.path.append('../..')

from shared.base_agent import BaseAgent, AgentResponse

logger = logging.getLogger(__name__)


class AdaptationAgent(BaseAgent):
    """
    Agent for adapting learning experience to student needs
    """
    
    def __init__(self):
        super().__init__(
            name="Adaptation Agent",
            model="claude-sonnet-4-20250514",
            temperature=0.5
        )
    
    def _get_system_prompt(self) -> str:
        """System prompt for adaptation"""
        return """You are an expert at adapting educational content to student needs.

Your role:
1. Analyze student performance patterns
2. Adjust content difficulty appropriately
3. Modify presentation style based on effectiveness
4. Identify optimal pacing for individual students
5. Recognize when to provide more scaffolding
6. Determine when student is ready to advance

Adaptation strategies:
- If accuracy > 90% and fast completion: Increase difficulty
- If accuracy 70-90%: Maintain current level, add variety
- If accuracy 50-70%: Provide more examples and practice
- If accuracy < 50%: Decrease difficulty, add scaffolding
- If slow but accurate: Reduce complexity, not difficulty
- If fast but inaccurate: Slow down, add reflection prompts

Scaffolding levels:
1. Full independence: Minimal guidance
2. Light scaffolding: Hints available
3. Medium scaffolding: Step-by-step guidance
4. Heavy scaffolding: Worked examples with explanation

Output format:
- Current performance analysis
- Recommended difficulty adjustment
- Suggested scaffolding level
- Content modifications needed
- Pacing recommendations"""
    
    async def process(self, input_data: Dict[str, Any]) -> AgentResponse:
        """
        Process adaptation request
        
        Args:
            input_data: {
                "student_id": str,
                "topic": str,
                "performance_history": List[Dict],
                "current_difficulty": str,
                "time_spent": float,
                "errors": List[Dict]
            }
        
        Returns:
            AgentResponse with adaptation recommendations
        """
        
        student_id = input_data.get("student_id")
        topic = input_data.get("topic")
        performance_history = input_data.get("performance_history", [])
        current_difficulty = input_data.get("current_difficulty", "medium")
        time_spent = input_data.get("time_spent", 0)
        errors = input_data.get("errors", [])
        
        if not student_id or not topic:
            raise ValueError("student_id and topic are required")
        
        logger.info(f"Analyzing adaptation needs for {student_id} on {topic}")
        
        # Analyze performance
        performance_analysis = self._analyze_performance(
            performance_history=performance_history,
            time_spent=time_spent,
            errors=errors
        )
        
        # Determine adaptations
        adaptations = self._determine_adaptations(
            performance_analysis=performance_analysis,
            current_difficulty=current_difficulty
        )
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            topic=topic,
            performance_analysis=performance_analysis,
            adaptations=adaptations
        )
        
        return AgentResponse(
            content=recommendations,
            agent_name=self.name,
            timestamp=datetime.utcnow(),
            metadata={
                "student_id": student_id,
                "topic": topic,
                "performance_analysis": performance_analysis,
                "adaptations": adaptations
            },
            confidence=0.85
        )
    
    def _analyze_performance(
        self,
        performance_history: List[Dict[str, Any]],
        time_spent: float,
        errors: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze student performance patterns"""
        
        if not performance_history:
            return {
                "accuracy": 0.0,
                "trend": "unknown",
                "speed": "unknown",
                "consistency": 0.0,
                "error_patterns": []
            }
        
        # Calculate accuracy
        correct = sum(1 for p in performance_history if p.get("correct", False))
        accuracy = correct / len(performance_history)
        
        # Analyze trend (improving, stable, declining)
        if len(performance_history) >= 3:
            recent_accuracy = sum(1 for p in performance_history[-3:] if p.get("correct", False)) / 3
            earlier_accuracy = sum(1 for p in performance_history[:3] if p.get("correct", False)) / 3
            
            if recent_accuracy > earlier_accuracy + 0.15:
                trend = "improving"
            elif recent_accuracy < earlier_accuracy - 0.15:
                trend = "declining"
            else:
                trend = "stable"
        else:
            trend = "insufficient_data"
        
        # Analyze speed
        if time_spent > 0:
            avg_time_per_problem = time_spent / len(performance_history)
            # Simplified - in production, compare to benchmarks
            if avg_time_per_problem < 60:
                speed = "fast"
            elif avg_time_per_problem < 180:
                speed = "moderate"
            else:
                speed = "slow"
        else:
            speed = "unknown"
        
        # Analyze consistency
        if len(performance_history) > 1:
            import statistics
            scores = [1.0 if p.get("correct") else 0.0 for p in performance_history]
            consistency = 1.0 - statistics.stdev(scores) if len(scores) > 1 else 1.0
        else:
            consistency = 1.0
        
        # Analyze error patterns
        error_patterns = self._identify_error_patterns(errors)
        
        return {
            "accuracy": round(accuracy, 3),
            "trend": trend,
            "speed": speed,
            "consistency": round(consistency, 3),
            "error_patterns": error_patterns,
            "total_attempts": len(performance_history),
            "correct_count": correct
        }
    
    def _identify_error_patterns(
        self,
        errors: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Identify common error patterns"""
        
        if not errors:
            return []
        
        # Group errors by type
        from collections import defaultdict
        error_types = defaultdict(int)
        
        for error in errors:
            error_type = error.get("type", "unknown")
            error_types[error_type] += 1
        
        # Sort by frequency
        patterns = [
            {"type": error_type, "count": count}
            for error_type, count in sorted(
                error_types.items(),
                key=lambda x: x[1],
                reverse=True
            )
        ]
        
        return patterns[:5]  # Top 5 patterns
    
    def _determine_adaptations(
        self,
        performance_analysis: Dict[str, Any],
        current_difficulty: str
    ) -> Dict[str, Any]:
        """Determine what adaptations to make"""
        
        accuracy = performance_analysis["accuracy"]
        trend = performance_analysis["trend"]
        speed = performance_analysis["speed"]
        
        # Difficulty adjustment
        if accuracy > 0.9 and speed == "fast":
            difficulty_adjustment = "increase"
            new_difficulty = self._adjust_difficulty_level(current_difficulty, 1)
        elif accuracy < 0.5 or (accuracy < 0.7 and trend == "declining"):
            difficulty_adjustment = "decrease"
            new_difficulty = self._adjust_difficulty_level(current_difficulty, -1)
        else:
            difficulty_adjustment = "maintain"
            new_difficulty = current_difficulty
        
        # Scaffolding level
        if accuracy >= 0.8:
            scaffolding = "minimal"
        elif accuracy >= 0.6:
            scaffolding = "light"
        elif accuracy >= 0.4:
            scaffolding = "medium"
        else:
            scaffolding = "heavy"
        
        # Pacing
        if speed == "fast" and accuracy > 0.8:
            pacing = "accelerated"
        elif speed == "slow" and accuracy < 0.6:
            pacing = "slow_down"
        else:
            pacing = "maintain"
        
        # Content modifications
        modifications = []
        
        if accuracy < 0.6:
            modifications.append("add_more_examples")
            modifications.append("break_into_smaller_steps")
        
        if speed == "slow":
            modifications.append("simplify_language")
            modifications.append("reduce_problem_length")
        
        if performance_analysis["error_patterns"]:
            modifications.append("address_specific_misconceptions")
        
        return {
            "difficulty_adjustment": difficulty_adjustment,
            "new_difficulty": new_difficulty,
            "scaffolding_level": scaffolding,
            "pacing": pacing,
            "content_modifications": modifications
        }
    
    def _adjust_difficulty_level(
        self,
        current_difficulty: str,
        adjustment: int
    ) -> str:
        """Adjust difficulty level up or down"""
        
        levels = ["beginner", "easy", "medium", "hard", "advanced", "expert"]
        
        try:
            current_index = levels.index(current_difficulty)
        except ValueError:
            current_index = 2  # Default to medium
        
        new_index = max(0, min(len(levels) - 1, current_index + adjustment))
        
        return levels[new_index]
    
    def _generate_recommendations(
        self,
        topic: str,
        performance_analysis: Dict[str, Any],
        adaptations: Dict[str, Any]
    ) -> str:
        """Generate human-readable recommendations"""
        
        parts = [
            f"Adaptation Recommendations: {topic}",
            f"\nPerformance Analysis:",
            f"  Accuracy: {performance_analysis['accuracy']:.1%}",
            f"  Trend: {performance_analysis['trend'].replace('_', ' ').title()}",
            f"  Speed: {performance_analysis['speed'].title()}",
            f"  Consistency: {performance_analysis['consistency']:.1%}",
        ]
        
        if performance_analysis["error_patterns"]:
            parts.append("\nCommon Error Patterns:")
            for pattern in performance_analysis["error_patterns"][:3]:
                parts.append(f"  • {pattern['type']}: {pattern['count']} occurrences")
        
        parts.append(f"\nRecommended Adaptations:")
        parts.append(f"  Difficulty: {adaptations['difficulty_adjustment'].title()} → {adaptations['new_difficulty']}")
        parts.append(f"  Scaffolding: {adaptations['scaffolding_level'].title()}")
        parts.append(f"  Pacing: {adaptations['pacing'].replace('_', ' ').title()}")
        
        if adaptations["content_modifications"]:
            parts.append("\nContent Modifications:")
            for mod in adaptations["content_modifications"]:
                parts.append(f"  • {mod.replace('_', ' ').title()}")
        
        return "\n".join(parts)