"""
Decision Quality

Evaluates quality of automated decisions and remediation actions
"""

from typing import Dict, Any, List, Optional
import logging
from datetime import datetime, timedelta
from collections import defaultdict

logger = logging.getLogger(__name__)


class DecisionQuality:
    """
    Evaluates quality of automated decisions
    """
    
    def __init__(self):
        # Decision records
        self.decisions: List[Dict[str, Any]] = []
    
    def record_decision(
        self,
        decision_id: str,
        decision_type: str,
        made_at: datetime,
        action_taken: str,
        outcome: Optional[str] = None,
        was_correct: Optional[bool] = None,
        confidence: float = 0.5,
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Record a decision
        
        Args:
            decision_id: Unique decision identifier
            decision_type: Type of decision (remediation, scaling, etc.)
            made_at: When decision was made
            action_taken: Action that was taken
            outcome: Outcome of the decision
            was_correct: Whether decision was correct
            confidence: Confidence score (0.0 to 1.0)
            context: Additional context
        """
        
        decision = {
            "id": decision_id,
            "type": decision_type,
            "made_at": made_at,
            "action_taken": action_taken,
            "outcome": outcome,
            "was_correct": was_correct,
            "confidence": confidence,
            "context": context or {}
        }
        
        self.decisions.append(decision)
        
        logger.info(f"Recorded decision: {decision_id} ({decision_type})")
    
    def mark_decision_outcome(
        self,
        decision_id: str,
        was_correct: bool,
        outcome: str
    ) -> None:
        """
        Mark the outcome of a decision
        
        Args:
            decision_id: Decision identifier
            was_correct: Whether decision was correct
            outcome: Description of outcome
        """
        
        for decision in self.decisions:
            if decision["id"] == decision_id:
                decision["was_correct"] = was_correct
                decision["outcome"] = outcome
                logger.info(
                    f"Marked decision {decision_id}: "
                    f"{'correct' if was_correct else 'incorrect'}"
                )
                return
        
        logger.warning(f"Decision not found: {decision_id}")
    
    def calculate_decision_accuracy(
        self,
        decision_type: Optional[str] = None,
        timeframe: Optional[timedelta] = None
    ) -> Optional[float]:
        """
        Calculate decision accuracy rate
        
        Args:
            decision_type: Filter by decision type
            timeframe: Only include decisions within timeframe
        
        Returns:
            Accuracy rate (0.0 to 1.0)
        """
        
        filtered = self._filter_decisions(
            decision_type=decision_type,
            timeframe=timeframe
        )
        
        # Only include evaluated decisions
        evaluated = [
            d for d in filtered
            if d["was_correct"] is not None
        ]
        
        if not evaluated:
            return None
        
        correct_decisions = sum(
            1 for d in evaluated
            if d["was_correct"] is True
        )
        
        accuracy = correct_decisions / len(evaluated)
        
        logger.info(
            f"Decision accuracy: {accuracy:.2%} "
            f"({correct_decisions}/{len(evaluated)} correct)"
        )
        
        return accuracy
    
    def calculate_accuracy_by_type(self) -> Dict[str, float]:
        """Calculate accuracy broken down by decision type"""
        
        accuracy_by_type = {}
        
        types = set(d["type"] for d in self.decisions)
        
        for decision_type in types:
            accuracy = self.calculate_decision_accuracy(decision_type=decision_type)
            if accuracy is not None:
                accuracy_by_type[decision_type] = accuracy
        
        return accuracy_by_type
    
    def calculate_confidence_calibration(self) -> Dict[str, Any]:
        """
        Calculate how well confidence scores align with actual accuracy
        
        Returns:
            Calibration metrics
        """
        
        # Group decisions by confidence buckets
        buckets = {
            "low (0.0-0.3)": [],
            "medium (0.3-0.7)": [],
            "high (0.7-1.0)": []
        }
        
        for decision in self.decisions:
            if decision["was_correct"] is None:
                continue
            
            confidence = decision["confidence"]
            
            if confidence < 0.3:
                buckets["low (0.0-0.3)"].append(decision)
            elif confidence < 0.7:
                buckets["medium (0.3-0.7)"].append(decision)
            else:
                buckets["high (0.7-1.0)"].append(decision)
        
        # Calculate accuracy for each bucket
        calibration = {}
        
        for bucket_name, decisions in buckets.items():
            if not decisions:
                continue
            
            correct = sum(1 for d in decisions if d["was_correct"])
            accuracy = correct / len(decisions)
            
            calibration[bucket_name] = {
                "count": len(decisions),
                "accuracy": accuracy,
                "expected_accuracy": self._get_expected_accuracy(bucket_name)
            }
        
        return calibration
    
    def _get_expected_accuracy(self, bucket_name: str) -> float:
        """Get expected accuracy for confidence bucket"""
        
        if "low" in bucket_name:
            return 0.15
        elif "medium" in bucket_name:
            return 0.5
        else:
            return 0.85
    
    def get_decision_impact(
        self,
        decision_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze impact of decisions
        
        Args:
            decision_type: Filter by decision type
        
        Returns:
            Impact analysis
        """
        
        filtered = self._filter_decisions(decision_type=decision_type)
        
        # Count outcomes
        outcomes = defaultdict(int)
        for decision in filtered:
            if decision["outcome"]:
                outcomes[decision["outcome"]] += 1
        
        # Analyze actions
        actions = defaultdict(int)
        for decision in filtered:
            actions[decision["action_taken"]] += 1
        
        return {
            "total_decisions": len(filtered),
            "outcomes": dict(outcomes),
            "actions": dict(actions),
            "most_common_action": max(actions.items(), key=lambda x: x[1])[0] if actions else None
        }
    
    def calculate_improvement_over_time(
        self,
        window_days: int = 7
    ) -> List[Dict[str, Any]]:
        """
        Calculate accuracy improvement over time
        
        Args:
            window_days: Size of rolling window
        
        Returns:
            Time series of accuracy
        """
        
        if not self.decisions:
            return []
        
        # Sort by time
        sorted_decisions = sorted(self.decisions, key=lambda d: d["made_at"])
        
        # Calculate accuracy in rolling windows
        time_series = []
        window = timedelta(days=window_days)
        
        for i in range(0, len(sorted_decisions), max(1, window_days)):
            end_time = sorted_decisions[min(i + window_days, len(sorted_decisions) - 1)]["made_at"]
            start_time = end_time - window
            
            window_decisions = [
                d for d in sorted_decisions
                if start_time <= d["made_at"] <= end_time
                and d["was_correct"] is not None
            ]
            
            if window_decisions:
                correct = sum(1 for d in window_decisions if d["was_correct"])
                accuracy = correct / len(window_decisions)
                
                time_series.append({
                    "end_date": end_time.isoformat(),
                    "accuracy": accuracy,
                    "decisions": len(window_decisions)
                })
        
        return time_series
    
    def _filter_decisions(
        self,
        decision_type: Optional[str] = None,
        timeframe: Optional[timedelta] = None
    ) -> List[Dict[str, Any]]:
        """Filter decisions by criteria"""
        
        filtered = self.decisions.copy()
        
        if decision_type:
            filtered = [
                d for d in filtered
                if d["type"] == decision_type
            ]
        
        if timeframe:
            cutoff = datetime.utcnow() - timeframe
            filtered = [
                d for d in filtered
                if d["made_at"] >= cutoff
            ]
        
        return filtered
    
    def get_summary(self) -> Dict[str, Any]:
        """Get comprehensive decision quality summary"""
        
        return {
            "total_decisions": len(self.decisions),
            "accuracy_overall": self.calculate_decision_accuracy(),
            "accuracy_by_type": self.calculate_accuracy_by_type(),
            "confidence_calibration": self.calculate_confidence_calibration(),
            "recent_impact": self.get_decision_impact(),
            "improvement_trend": self.calculate_improvement_over_time()
        }