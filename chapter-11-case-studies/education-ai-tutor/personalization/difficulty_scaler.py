"""
Difficulty Scaler

Dynamically scales difficulty based on student performance
"""

from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)


class DifficultyScaler:
    """
    Scales difficulty dynamically based on performance
    """
    
    def __init__(self):
        # Performance tracking for scaling
        self.performance_history: Dict[str, List[Dict[str, Any]]] = {}
        
        # Scaling parameters
        self.target_accuracy = 0.75  # Target 75% accuracy
        self.adjustment_sensitivity = 0.15  # How aggressively to adjust
    
    def calculate_next_difficulty(
        self,
        student_id: str,
        current_difficulty: str,
        recent_performance: List[Dict[str, Any]],
        topic: str
    ) -> Dict[str, Any]:
        """
        Calculate appropriate next difficulty level
        
        Args:
            student_id: Student identifier
            current_difficulty: Current difficulty level
            recent_performance: Recent performance data
            topic: Topic being studied
        
        Returns:
            Difficulty recommendation with reasoning
        """
        
        if not recent_performance:
            return {
                "recommended_difficulty": current_difficulty,
                "adjustment": "maintain",
                "confidence": 0.5,
                "reason": "No performance data available"
            }
        
        # Analyze performance
        analysis = self._analyze_performance(recent_performance)
        
        # Determine adjustment
        adjustment = self._determine_adjustment(
            analysis=analysis,
            current_difficulty=current_difficulty
        )
        
        # Record for history
        self._record_scaling_decision(
            student_id=student_id,
            topic=topic,
            current_difficulty=current_difficulty,
            analysis=analysis,
            adjustment=adjustment
        )
        
        return adjustment
    
    def _analyze_performance(
        self,
        performance: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze performance metrics"""
        
        # Calculate metrics
        total = len(performance)
        correct = sum(1 for p in performance if p.get("correct", False))
        accuracy = correct / total if total > 0 else 0
        
        # Analyze time taken
        times = [p.get("time_taken", 60) for p in performance]
        avg_time = sum(times) / len(times)
        
        # Analyze attempts
        attempts = [p.get("attempts", 1) for p in performance]
        avg_attempts = sum(attempts) / len(attempts)
        
        # Analyze confidence/certainty
        certainties = [p.get("certainty", 0.5) for p in performance if "certainty" in p]
        avg_certainty = sum(certainties) / len(certainties) if certainties else 0.5
        
        # Trend analysis
        if total >= 3:
            recent_accuracy = sum(1 for p in performance[-3:] if p.get("correct")) / 3
            earlier_accuracy = sum(1 for p in performance[:3] if p.get("correct")) / 3
            trend = "improving" if recent_accuracy > earlier_accuracy else \
                   "declining" if recent_accuracy < earlier_accuracy else "stable"
        else:
            trend = "unknown"
        
        return {
            "accuracy": accuracy,
            "avg_time": avg_time,
            "avg_attempts": avg_attempts,
            "avg_certainty": avg_certainty,
            "trend": trend,
            "sample_size": total
        }
    
    def _determine_adjustment(
        self,
        analysis: Dict[str, Any],
        current_difficulty: str
    ) -> Dict[str, Any]:
        """Determine difficulty adjustment"""
        
        accuracy = analysis["accuracy"]
        trend = analysis["trend"]
        avg_time = analysis["avg_time"]
        avg_certainty = analysis["avg_certainty"]
        
        difficulty_levels = ["beginner", "easy", "medium", "hard", "advanced", "expert"]
        
        try:
            current_index = difficulty_levels.index(current_difficulty)
        except ValueError:
            current_index = 2  # Default to medium
        
        # Decision logic
        adjustment_amount = 0
        reasons = []
        
        # Accuracy-based adjustment
        if accuracy > 0.9 and avg_certainty > 0.7:
            adjustment_amount += 1
            reasons.append(f"High accuracy ({accuracy:.1%}) with confidence")
        elif accuracy > 0.85:
            adjustment_amount += 1
            reasons.append(f"Consistently high accuracy ({accuracy:.1%})")
        elif accuracy < 0.5:
            adjustment_amount -= 1
            reasons.append(f"Low accuracy ({accuracy:.1%})")
        elif accuracy < 0.6 and trend == "declining":
            adjustment_amount -= 1
            reasons.append(f"Declining performance ({accuracy:.1%})")
        
        # Time-based adjustment
        expected_time = 60  # seconds
        if avg_time < expected_time * 0.6 and accuracy > 0.8:
            # Fast and accurate
            reasons.append("Completing quickly with high accuracy")
        elif avg_time > expected_time * 2:
            adjustment_amount -= 1
            reasons.append("Taking significantly longer than expected")
        
        # Apply adjustment
        new_index = max(0, min(len(difficulty_levels) - 1, current_index + adjustment_amount))
        new_difficulty = difficulty_levels[new_index]
        
        # Determine adjustment type
        if new_index > current_index:
            adjustment_type = "increase"
        elif new_index < current_index:
            adjustment_type = "decrease"
        else:
            adjustment_type = "maintain"
        
        # Calculate confidence in recommendation
        confidence = self._calculate_confidence(analysis)
        
        return {
            "recommended_difficulty": new_difficulty,
            "current_difficulty": current_difficulty,
            "adjustment": adjustment_type,
            "adjustment_amount": abs(adjustment_amount),
            "confidence": confidence,
            "reason": " | ".join(reasons) if reasons else "Performance appropriate for current level",
            "analysis": analysis
        }
    
    def _calculate_confidence(self, analysis: Dict[str, Any]) -> float:
        """Calculate confidence in recommendation"""
        
        # Higher confidence with:
        # - More data points
        # - Consistent performance
        # - Clear trends
        
        confidence = 0.5  # Base
        
        # Sample size factor
        sample_size = analysis["sample_size"]
        if sample_size >= 10:
            confidence += 0.2
        elif sample_size >= 5:
            confidence += 0.1
        
        # Trend clarity
        if analysis["trend"] in ["improving", "declining"]:
            confidence += 0.15
        
        # Performance clarity (not borderline)
        accuracy = analysis["accuracy"]
        if accuracy > 0.85 or accuracy < 0.55:
            confidence += 0.15
        
        return min(1.0, confidence)
    
    def _record_scaling_decision(
        self,
        student_id: str,
        topic: str,
        current_difficulty: str,
        analysis: Dict[str, Any],
        adjustment: Dict[str, Any]
    ) -> None:
        """Record scaling decision for analysis"""
        
        key = f"{student_id}_{topic}"
        
        if key not in self.performance_history:
            self.performance_history[key] = []
        
        self.performance_history[key].append({
            "timestamp": datetime.now().isoformat(),
            "current_difficulty": current_difficulty,
            "recommended_difficulty": adjustment["recommended_difficulty"],
            "adjustment": adjustment["adjustment"],
            "analysis": analysis
        })
        
        # Keep only recent history
        self.performance_history[key] = self.performance_history[key][-50:]
    
    def get_optimal_difficulty_distribution(
        self,
        mastery_level: float
    ) -> Dict[str, float]:
        """
        Get optimal distribution of difficulty levels for practice
        
        Args:
            mastery_level: Student's mastery level (0.0 to 1.0)
        
        Returns:
            Distribution of difficulties
        """
        
        # Zone of Proximal Development approach
        # Most content at student's level, some easier, some harder
        
        if mastery_level < 0.3:
            return {
                "beginner": 0.6,
                "easy": 0.3,
                "medium": 0.1
            }
        elif mastery_level < 0.5:
            return {
                "beginner": 0.1,
                "easy": 0.5,
                "medium": 0.3,
                "hard": 0.1
            }
        elif mastery_level < 0.7:
            return {
                "easy": 0.1,
                "medium": 0.5,
                "hard": 0.3,
                "advanced": 0.1
            }
        else:
            return {
                "medium": 0.1,
                "hard": 0.4,
                "advanced": 0.4,
                "expert": 0.1
            }