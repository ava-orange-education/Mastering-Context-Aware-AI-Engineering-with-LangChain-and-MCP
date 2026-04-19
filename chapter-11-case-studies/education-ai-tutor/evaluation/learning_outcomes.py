"""
Learning Outcomes Evaluator

Evaluates actual learning outcomes and knowledge gains
"""

from typing import Dict, Any, List, Optional
import logging
from datetime import datetime, timedelta
import statistics

logger = logging.getLogger(__name__)


class LearningOutcomesEvaluator:
    """
    Evaluates learning outcomes and knowledge gains
    """
    
    def __init__(self):
        # Assessment data
        self.pre_assessments: Dict[str, Dict[str, Any]] = {}
        self.post_assessments: Dict[str, Dict[str, Any]] = {}
        
        # Progress data
        self.progress_snapshots: Dict[str, List[Dict[str, Any]]] = {}
    
    def record_pre_assessment(
        self,
        student_id: str,
        topic: str,
        assessment_results: Dict[str, Any]
    ) -> None:
        """
        Record pre-assessment (baseline knowledge)
        
        Args:
            student_id: Student identifier
            topic: Topic assessed
            assessment_results: Assessment results
        """
        
        key = f"{student_id}_{topic}"
        
        self.pre_assessments[key] = {
            "student_id": student_id,
            "topic": topic,
            "timestamp": datetime.utcnow(),
            "score": assessment_results.get("score", 0),
            "mastery_level": assessment_results.get("mastery_level", 0),
            "details": assessment_results
        }
        
        logger.info(f"Recorded pre-assessment for {student_id} on {topic}: {assessment_results.get('score', 0):.1%}")
    
    def record_post_assessment(
        self,
        student_id: str,
        topic: str,
        assessment_results: Dict[str, Any]
    ) -> None:
        """
        Record post-assessment (learning outcome)
        
        Args:
            student_id: Student identifier
            topic: Topic assessed
            assessment_results: Assessment results
        """
        
        key = f"{student_id}_{topic}"
        
        self.post_assessments[key] = {
            "student_id": student_id,
            "topic": topic,
            "timestamp": datetime.utcnow(),
            "score": assessment_results.get("score", 0),
            "mastery_level": assessment_results.get("mastery_level", 0),
            "details": assessment_results
        }
        
        logger.info(f"Recorded post-assessment for {student_id} on {topic}: {assessment_results.get('score', 0):.1%}")
    
    def calculate_learning_gain(
        self,
        student_id: str,
        topic: str
    ) -> Dict[str, Any]:
        """
        Calculate learning gain between pre and post assessments
        
        Args:
            student_id: Student identifier
            topic: Topic to evaluate
        
        Returns:
            Learning gain metrics
        """
        
        key = f"{student_id}_{topic}"
        
        pre = self.pre_assessments.get(key)
        post = self.post_assessments.get(key)
        
        if not pre or not post:
            return {
                "available": False,
                "message": "Both pre and post assessments required"
            }
        
        # Calculate absolute gain
        absolute_gain = post["score"] - pre["score"]
        
        # Calculate normalized gain (Hake gain)
        # g = (post - pre) / (1 - pre)
        if pre["score"] < 1.0:
            normalized_gain = (post["score"] - pre["score"]) / (1.0 - pre["score"])
        else:
            normalized_gain = 0.0
        
        # Determine gain level
        if normalized_gain >= 0.7:
            gain_level = "high"
        elif normalized_gain >= 0.3:
            gain_level = "medium"
        elif normalized_gain > 0:
            gain_level = "low"
        else:
            gain_level = "none"
        
        # Calculate mastery improvement
        mastery_gain = post["mastery_level"] - pre["mastery_level"]
        
        return {
            "available": True,
            "student_id": student_id,
            "topic": topic,
            "pre_score": round(pre["score"], 3),
            "post_score": round(post["score"], 3),
            "absolute_gain": round(absolute_gain, 3),
            "normalized_gain": round(normalized_gain, 3),
            "gain_level": gain_level,
            "mastery_gain": round(mastery_gain, 3),
            "time_between": (post["timestamp"] - pre["timestamp"]).days,
            "pre_assessment_date": pre["timestamp"].isoformat(),
            "post_assessment_date": post["timestamp"].isoformat()
        }
    
    def evaluate_cohort_outcomes(
        self,
        topic: str,
        student_ids: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate learning outcomes for a cohort
        
        Args:
            topic: Topic to evaluate
            student_ids: Optional list of specific students
        
        Returns:
            Cohort-level outcomes
        """
        
        # Find all students with both pre and post assessments
        gains = []
        
        for key, pre in self.pre_assessments.items():
            if topic not in key:
                continue
            
            student_id = pre["student_id"]
            
            if student_ids and student_id not in student_ids:
                continue
            
            if key in self.post_assessments:
                gain = self.calculate_learning_gain(student_id, topic)
                if gain["available"]:
                    gains.append(gain)
        
        if not gains:
            return {
                "available": False,
                "message": "No complete assessment pairs found"
            }
        
        # Calculate cohort statistics
        absolute_gains = [g["absolute_gain"] for g in gains]
        normalized_gains = [g["normalized_gain"] for g in gains]
        
        return {
            "available": True,
            "topic": topic,
            "students_evaluated": len(gains),
            "mean_absolute_gain": round(statistics.mean(absolute_gains), 3),
            "median_absolute_gain": round(statistics.median(absolute_gains), 3),
            "std_absolute_gain": round(statistics.stdev(absolute_gains), 3) if len(absolute_gains) > 1 else 0,
            "mean_normalized_gain": round(statistics.mean(normalized_gains), 3),
            "median_normalized_gain": round(statistics.median(normalized_gains), 3),
            "high_gain_count": sum(1 for g in gains if g["gain_level"] == "high"),
            "medium_gain_count": sum(1 for g in gains if g["gain_level"] == "medium"),
            "low_gain_count": sum(1 for g in gains if g["gain_level"] == "low"),
            "no_gain_count": sum(1 for g in gains if g["gain_level"] == "none"),
            "individual_gains": gains
        }
    
    def track_progress_over_time(
        self,
        student_id: str,
        topic: str,
        current_mastery: float
    ) -> None:
        """
        Record progress snapshot
        
        Args:
            student_id: Student identifier
            topic: Topic
            current_mastery: Current mastery level
        """
        
        key = f"{student_id}_{topic}"
        
        if key not in self.progress_snapshots:
            self.progress_snapshots[key] = []
        
        self.progress_snapshots[key].append({
            "timestamp": datetime.utcnow(),
            "mastery": current_mastery
        })
    
    def analyze_learning_trajectory(
        self,
        student_id: str,
        topic: str
    ) -> Dict[str, Any]:
        """
        Analyze learning trajectory over time
        
        Args:
            student_id: Student identifier
            topic: Topic to analyze
        
        Returns:
            Trajectory analysis
        """
        
        key = f"{student_id}_{topic}"
        snapshots = self.progress_snapshots.get(key, [])
        
        if len(snapshots) < 2:
            return {
                "available": False,
                "message": "Need at least 2 snapshots for trajectory analysis"
            }
        
        # Sort by timestamp
        snapshots.sort(key=lambda x: x["timestamp"])
        
        # Calculate rate of learning
        first = snapshots[0]
        last = snapshots[-1]
        
        time_diff = (last["timestamp"] - first["timestamp"]).total_seconds() / 3600  # hours
        mastery_diff = last["mastery"] - first["mastery"]
        
        learning_rate = mastery_diff / time_diff if time_diff > 0 else 0
        
        # Analyze trend
        masteries = [s["mastery"] for s in snapshots]
        
        # Simple linear regression
        n = len(masteries)
        x = list(range(n))
        y = masteries
        
        x_mean = sum(x) / n
        y_mean = sum(y) / n
        
        numerator = sum((x[i] - x_mean) * (y[i] - y_mean) for i in range(n))
        denominator = sum((x[i] - x_mean) ** 2 for i in range(n))
        
        slope = numerator / denominator if denominator != 0 else 0
        
        # Determine trend
        if slope > 0.05:
            trend = "improving"
        elif slope < -0.05:
            trend = "declining"
        else:
            trend = "stable"
        
        return {
            "available": True,
            "student_id": student_id,
            "topic": topic,
            "snapshots": len(snapshots),
            "start_mastery": round(first["mastery"], 3),
            "current_mastery": round(last["mastery"], 3),
            "total_gain": round(mastery_diff, 3),
            "time_span_hours": round(time_diff, 1),
            "learning_rate_per_hour": round(learning_rate, 4),
            "trend": trend,
            "slope": round(slope, 4),
            "mastery_history": [round(m, 3) for m in masteries]
        }
    
    def evaluate_retention(
        self,
        student_id: str,
        topic: str,
        weeks_after: int = 4
    ) -> Dict[str, Any]:
        """
        Evaluate knowledge retention after time period
        
        Args:
            student_id: Student identifier
            topic: Topic to evaluate
            weeks_after: Weeks after initial learning
        
        Returns:
            Retention analysis
        """
        
        key = f"{student_id}_{topic}"
        
        # Get post-assessment (end of learning)
        post = self.post_assessments.get(key)
        
        if not post:
            return {
                "available": False,
                "message": "No post-assessment found"
            }
        
        # Find mastery level after specified weeks
        target_date = post["timestamp"] + timedelta(weeks=weeks_after)
        
        snapshots = self.progress_snapshots.get(key, [])
        
        # Find snapshot closest to target date
        retention_snapshot = None
        min_diff = float('inf')
        
        for snapshot in snapshots:
            if snapshot["timestamp"] >= target_date:
                diff = abs((snapshot["timestamp"] - target_date).days)
                if diff < min_diff:
                    min_diff = diff
                    retention_snapshot = snapshot
        
        if not retention_snapshot:
            return {
                "available": False,
                "message": f"No snapshot found around {weeks_after} weeks after completion"
            }
        
        # Calculate retention
        initial_mastery = post["mastery_level"]
        retained_mastery = retention_snapshot["mastery"]
        
        retention_rate = retained_mastery / initial_mastery if initial_mastery > 0 else 0
        
        # Classify retention
        if retention_rate >= 0.9:
            retention_level = "excellent"
        elif retention_rate >= 0.75:
            retention_level = "good"
        elif retention_rate >= 0.6:
            retention_level = "moderate"
        else:
            retention_level = "poor"
        
        return {
            "available": True,
            "student_id": student_id,
            "topic": topic,
            "weeks_after": weeks_after,
            "initial_mastery": round(initial_mastery, 3),
            "retained_mastery": round(retained_mastery, 3),
            "retention_rate": round(retention_rate, 3),
            "retention_level": retention_level,
            "mastery_loss": round(initial_mastery - retained_mastery, 3),
            "completion_date": post["timestamp"].isoformat(),
            "retention_check_date": retention_snapshot["timestamp"].isoformat()
        }