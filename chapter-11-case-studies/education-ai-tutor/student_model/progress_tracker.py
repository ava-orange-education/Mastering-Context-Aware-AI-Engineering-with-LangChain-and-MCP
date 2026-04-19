"""
Progress Tracker

Tracks student progress over time across subjects and skills
"""

from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


class ProgressTracker:
    """
    Tracks and analyzes student progress
    """
    
    def __init__(self, student_id: str):
        self.student_id = student_id
        
        # Session history
        self.sessions: List[Dict[str, Any]] = []
        
        # Performance by subject
        self.subject_performance: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        
        # Milestones achieved
        self.milestones: List[Dict[str, Any]] = []
        
        # Learning streaks
        self.current_streak = 0
        self.longest_streak = 0
        self.last_activity_date: Optional[datetime] = None
        
        # Time tracking
        self.total_time_spent = 0.0  # in minutes
        self.time_by_subject: Dict[str, float] = defaultdict(float)
        
        # Achievement tracking
        self.achievements: List[Dict[str, Any]] = []
        
        # Created timestamp
        self.created_at = datetime.utcnow()
    
    def record_session(
        self,
        subject: str,
        topic: str,
        duration: float,
        problems_attempted: int,
        problems_correct: int,
        concepts_covered: List[str],
        difficulty_level: str,
        notes: Optional[str] = None
    ) -> None:
        """
        Record a learning session
        
        Args:
            subject: Subject area
            topic: Specific topic
            duration: Session duration in minutes
            problems_attempted: Number of problems attempted
            problems_correct: Number of problems correct
            concepts_covered: List of concepts covered
            difficulty_level: Difficulty level
            notes: Optional session notes
        """
        
        session = {
            "session_id": f"session_{len(self.sessions) + 1}",
            "subject": subject,
            "topic": topic,
            "duration": duration,
            "problems_attempted": problems_attempted,
            "problems_correct": problems_correct,
            "accuracy": problems_correct / problems_attempted if problems_attempted > 0 else 0,
            "concepts_covered": concepts_covered,
            "difficulty_level": difficulty_level,
            "notes": notes,
            "timestamp": datetime.utcnow()
        }
        
        self.sessions.append(session)
        
        # Update subject performance
        self.subject_performance[subject].append({
            "accuracy": session["accuracy"],
            "duration": duration,
            "timestamp": session["timestamp"]
        })
        
        # Update time tracking
        self.total_time_spent += duration
        self.time_by_subject[subject] += duration
        
        # Update streak
        self._update_streak()
        
        # Check for milestones
        self._check_milestones(session)
        
        logger.info(
            f"Recorded session for {self.student_id}: {subject}/{topic} "
            f"({duration:.1f} min, {session['accuracy']:.1%} accuracy)"
        )
    
    def _update_streak(self) -> None:
        """Update learning streak"""
        
        today = datetime.utcnow().date()
        
        if self.last_activity_date:
            last_date = self.last_activity_date.date()
            days_diff = (today - last_date).days
            
            if days_diff == 0:
                # Same day, streak continues
                pass
            elif days_diff == 1:
                # Consecutive day
                self.current_streak += 1
                self.longest_streak = max(self.longest_streak, self.current_streak)
            else:
                # Streak broken
                self.current_streak = 1
        else:
            # First activity
            self.current_streak = 1
        
        self.last_activity_date = datetime.utcnow()
    
    def _check_milestones(self, session: Dict[str, Any]) -> None:
        """Check and record milestone achievements"""
        
        # Check various milestones
        milestones_to_check = []
        
        # Total sessions milestone
        if len(self.sessions) in [1, 10, 25, 50, 100]:
            milestones_to_check.append({
                "type": "sessions_completed",
                "description": f"Completed {len(self.sessions)} learning sessions",
                "value": len(self.sessions)
            })
        
        # Perfect session
        if session["accuracy"] == 1.0 and session["problems_attempted"] >= 5:
            milestones_to_check.append({
                "type": "perfect_session",
                "description": f"Perfect score on {session['topic']}",
                "value": 1.0
            })
        
        # Streak milestones
        if self.current_streak in [3, 7, 14, 30, 100]:
            milestones_to_check.append({
                "type": "learning_streak",
                "description": f"{self.current_streak}-day learning streak",
                "value": self.current_streak
            })
        
        # Time spent milestones (in hours)
        hours_spent = self.total_time_spent / 60
        if int(hours_spent) in [1, 10, 25, 50, 100] and hours_spent % 1 < 0.2:
            milestones_to_check.append({
                "type": "time_invested",
                "description": f"{int(hours_spent)} hours of learning",
                "value": hours_spent
            })
        
        # Add new milestones
        for milestone in milestones_to_check:
            # Check if already achieved
            if not any(m["type"] == milestone["type"] and m["value"] == milestone["value"] 
                      for m in self.milestones):
                milestone["achieved_at"] = datetime.utcnow()
                self.milestones.append(milestone)
                logger.info(f"Milestone achieved: {milestone['description']}")
    
    def get_progress_summary(self, days: int = 30) -> Dict[str, Any]:
        """
        Get progress summary for time period
        
        Args:
            days: Number of days to analyze
        
        Returns:
            Summary statistics
        """
        
        cutoff = datetime.utcnow() - timedelta(days=days)
        recent_sessions = [s for s in self.sessions if s["timestamp"] > cutoff]
        
        if not recent_sessions:
            return {
                "period_days": days,
                "total_sessions": 0,
                "message": f"No activity in the last {days} days"
            }
        
        # Calculate metrics
        total_problems = sum(s["problems_attempted"] for s in recent_sessions)
        correct_problems = sum(s["problems_correct"] for s in recent_sessions)
        total_time = sum(s["duration"] for s in recent_sessions)
        
        # Accuracy by subject
        subject_accuracy = {}
        for subject, performances in self.subject_performance.items():
            recent_perfs = [p for p in performances if p["timestamp"] > cutoff]
            if recent_perfs:
                avg_acc = sum(p["accuracy"] for p in recent_perfs) / len(recent_perfs)
                subject_accuracy[subject] = round(avg_acc, 3)
        
        return {
            "period_days": days,
            "total_sessions": len(recent_sessions),
            "total_time_minutes": round(total_time, 1),
            "total_time_hours": round(total_time / 60, 1),
            "total_problems_attempted": total_problems,
            "total_problems_correct": correct_problems,
            "overall_accuracy": round(correct_problems / total_problems, 3) if total_problems > 0 else 0,
            "current_streak": self.current_streak,
            "longest_streak": self.longest_streak,
            "subjects_practiced": len(subject_accuracy),
            "accuracy_by_subject": subject_accuracy,
            "recent_milestones": [m for m in self.milestones if m["achieved_at"] > cutoff]
        }
    
    def get_learning_velocity(self, days: int = 7) -> Dict[str, Any]:
        """
        Calculate learning velocity (rate of progress)
        
        Args:
            days: Period to analyze
        
        Returns:
            Velocity metrics
        """
        
        cutoff = datetime.utcnow() - timedelta(days=days)
        recent_sessions = [s for s in self.sessions if s["timestamp"] > cutoff]
        
        if len(recent_sessions) < 2:
            return {
                "insufficient_data": True,
                "message": "Need at least 2 sessions to calculate velocity"
            }
        
        # Calculate accuracy trend
        accuracies = [s["accuracy"] for s in recent_sessions]
        
        # Simple linear trend
        if len(accuracies) > 1:
            import statistics
            recent_avg = statistics.mean(accuracies[-5:]) if len(accuracies) >= 5 else statistics.mean(accuracies)
            earlier_avg = statistics.mean(accuracies[:5]) if len(accuracies) >= 5 else statistics.mean(accuracies[:len(accuracies)//2])
            
            trend = "improving" if recent_avg > earlier_avg + 0.05 else \
                   "declining" if recent_avg < earlier_avg - 0.05 else \
                   "stable"
        else:
            trend = "unknown"
        
        # Sessions per week
        sessions_per_week = len(recent_sessions) / (days / 7)
        
        # Average session duration
        avg_duration = sum(s["duration"] for s in recent_sessions) / len(recent_sessions)
        
        return {
            "trend": trend,
            "sessions_per_week": round(sessions_per_week, 1),
            "avg_session_duration": round(avg_duration, 1),
            "consistency_score": self._calculate_consistency(recent_sessions)
        }
    
    def _calculate_consistency(self, sessions: List[Dict[str, Any]]) -> float:
        """Calculate consistency score (0.0 to 1.0)"""
        
        if len(sessions) < 2:
            return 0.0
        
        # Check distribution across days
        dates = [s["timestamp"].date() for s in sessions]
        unique_dates = len(set(dates))
        
        # More days with activity = higher consistency
        days_active = unique_dates
        days_in_period = (max(dates) - min(dates)).days + 1
        
        consistency = days_active / days_in_period if days_in_period > 0 else 0
        
        return round(min(1.0, consistency), 3)
    
    def get_achievements(self) -> List[Dict[str, Any]]:
        """Get all achievements"""
        return self.achievements.copy()
    
    def add_achievement(
        self,
        name: str,
        description: str,
        category: str = "general"
    ) -> None:
        """Add a custom achievement"""
        
        achievement = {
            "name": name,
            "description": description,
            "category": category,
            "earned_at": datetime.utcnow()
        }
        
        self.achievements.append(achievement)
        logger.info(f"Achievement earned: {name}")