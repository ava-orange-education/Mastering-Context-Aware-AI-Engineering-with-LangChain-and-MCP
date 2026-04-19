"""
Difficulty Adapter

Adjusts content difficulty based on student performance
"""

from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)


class DifficultyAdapter:
    """
    Adapts content difficulty dynamically
    """
    
    def __init__(self):
        # Difficulty levels in order
        self.difficulty_levels = [
            "beginner",
            "easy",
            "medium",
            "hard",
            "advanced",
            "expert"
        ]
        
        # Performance thresholds for adjustments
        self.increase_threshold = 0.85  # Increase if accuracy > 85%
        self.decrease_threshold = 0.50  # Decrease if accuracy < 50%
    
    def recommend_difficulty(
        self,
        current_difficulty: str,
        recent_performance: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Recommend difficulty adjustment
        
        Args:
            current_difficulty: Current difficulty level
            recent_performance: List of recent performance data
        
        Returns:
            Recommendation with new difficulty and reasoning
        """
        
        if not recent_performance:
            return {
                "recommended_difficulty": current_difficulty,
                "change": "maintain",
                "reason": "Insufficient performance data"
            }
        
        # Calculate average accuracy
        accuracies = [p.get("accuracy", 0.5) for p in recent_performance]
        avg_accuracy = sum(accuracies) / len(accuracies)
        
        # Calculate average time taken (normalized)
        times = [p.get("time_taken", 60) for p in recent_performance]
        avg_time = sum(times) / len(times)
        expected_time = 60  # seconds per problem
        time_ratio = avg_time / expected_time
        
        # Determine adjustment
        if avg_accuracy >= self.increase_threshold and time_ratio < 1.2:
            # High accuracy and reasonable time -> increase difficulty
            new_difficulty = self._adjust_difficulty(current_difficulty, 1)
            change = "increase"
            reason = f"High accuracy ({avg_accuracy:.1%}) indicates readiness for harder content"
        
        elif avg_accuracy <= self.decrease_threshold:
            # Low accuracy -> decrease difficulty
            new_difficulty = self._adjust_difficulty(current_difficulty, -1)
            change = "decrease"
            reason = f"Low accuracy ({avg_accuracy:.1%}) suggests current level is too challenging"
        
        elif time_ratio > 2.0 and avg_accuracy < 0.7:
            # Taking too long and not performing well -> decrease
            new_difficulty = self._adjust_difficulty(current_difficulty, -1)
            change = "decrease"
            reason = "Struggling with time and accuracy"
        
        else:
            # Maintain current level
            new_difficulty = current_difficulty
            change = "maintain"
            reason = f"Performance ({avg_accuracy:.1%}) appropriate for current level"
        
        return {
            "recommended_difficulty": new_difficulty,
            "current_difficulty": current_difficulty,
            "change": change,
            "reason": reason,
            "avg_accuracy": round(avg_accuracy, 3),
            "avg_time": round(avg_time, 1),
            "performance_samples": len(recent_performance)
        }
    
    def _adjust_difficulty(self, current: str, adjustment: int) -> str:
        """
        Adjust difficulty level
        
        Args:
            current: Current difficulty
            adjustment: +1 to increase, -1 to decrease
        
        Returns:
            New difficulty level
        """
        
        try:
            current_index = self.difficulty_levels.index(current)
        except ValueError:
            # Unknown level, default to medium
            current_index = 2
        
        new_index = max(0, min(len(self.difficulty_levels) - 1, current_index + adjustment))
        
        return self.difficulty_levels[new_index]
    
    def get_difficulty_range(
        self,
        target_difficulty: str,
        range_size: int = 3
    ) -> List[str]:
        """
        Get a range of difficulties centered on target
        
        Args:
            target_difficulty: Target difficulty level
            range_size: Size of range (odd number)
        
        Returns:
            List of difficulty levels
        """
        
        try:
            target_index = self.difficulty_levels.index(target_difficulty)
        except ValueError:
            target_index = 2
        
        # Calculate range
        half_range = range_size // 2
        start_index = max(0, target_index - half_range)
        end_index = min(len(self.difficulty_levels), target_index + half_range + 1)
        
        return self.difficulty_levels[start_index:end_index]
    
    def calculate_adaptive_difficulty(
        self,
        student_mastery: float,
        topic_complexity: float = 0.5
    ) -> str:
        """
        Calculate appropriate difficulty based on mastery and topic complexity
        
        Args:
            student_mastery: Student's mastery level (0.0 to 1.0)
            topic_complexity: Topic's inherent complexity (0.0 to 1.0)
        
        Returns:
            Appropriate difficulty level
        """
        
        # Combine mastery and complexity
        # Lower complexity topics can be harder even with lower mastery
        # Higher complexity topics need higher mastery
        
        adjusted_level = student_mastery * (1.0 - topic_complexity * 0.3)
        
        # Map to difficulty level
        if adjusted_level < 0.2:
            return "beginner"
        elif adjusted_level < 0.4:
            return "easy"
        elif adjusted_level < 0.6:
            return "medium"
        elif adjusted_level < 0.75:
            return "hard"
        elif adjusted_level < 0.9:
            return "advanced"
        else:
            return "expert"
    
    def get_scaffolding_level(
        self,
        accuracy: float,
        attempts: int
    ) -> str:
        """
        Determine appropriate scaffolding level
        
        Args:
            accuracy: Recent accuracy
            attempts: Number of attempts on current problem
        
        Returns:
            Scaffolding level (none, light, medium, heavy)
        """
        
        if accuracy >= 0.8 and attempts == 1:
            return "none"
        elif accuracy >= 0.6 or attempts <= 2:
            return "light"
        elif accuracy >= 0.4 or attempts <= 3:
            return "medium"
        else:
            return "heavy"