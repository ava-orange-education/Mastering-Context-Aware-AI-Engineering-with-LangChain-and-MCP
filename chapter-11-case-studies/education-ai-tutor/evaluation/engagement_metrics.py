"""
Engagement Metrics

Tracks and analyzes student engagement
"""

from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


class EngagementMetrics:
    """
    Tracks and analyzes student engagement metrics
    """
    
    def __init__(self):
        # Session data
        self.sessions: List[Dict[str, Any]] = []
        
        # Interaction data
        self.interactions: List[Dict[str, Any]] = []
        
        # Content engagement
        self.content_engagement: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    
    def record_session(
        self,
        student_id: str,
        duration: float,
        topics_covered: List[str],
        problems_attempted: int,
        problems_completed: int,
        voluntary: bool = True
    ) -> None:
        """
        Record a learning session
        
        Args:
            student_id: Student identifier
            duration: Session duration in minutes
            topics_covered: Topics studied
            problems_attempted: Number of problems attempted
            problems_completed: Number of problems completed
            voluntary: Whether session was voluntary or required
        """
        
        session = {
            "student_id": student_id,
            "timestamp": datetime.utcnow(),
            "duration": duration,
            "topics_covered": topics_covered,
            "problems_attempted": problems_attempted,
            "problems_completed": problems_completed,
            "completion_rate": problems_completed / problems_attempted if problems_attempted > 0 else 0,
            "voluntary": voluntary
        }
        
        self.sessions.append(session)
        
        logger.info(
            f"Recorded session for {student_id}: {duration:.1f} min, "
            f"{problems_completed}/{problems_attempted} problems"
        )
    
    def record_interaction(
        self,
        student_id: str,
        interaction_type: str,
        content_id: str,
        time_spent: float,
        completed: bool,
        revisited: bool = False
    ) -> None:
        """
        Record content interaction
        
        Args:
            student_id: Student identifier
            interaction_type: Type of interaction (view, attempt, complete)
            content_id: Content identifier
            time_spent: Time spent in seconds
            completed: Whether interaction was completed
            revisited: Whether this content was revisited
        """
        
        interaction = {
            "student_id": student_id,
            "timestamp": datetime.utcnow(),
            "interaction_type": interaction_type,
            "content_id": content_id,
            "time_spent": time_spent,
            "completed": completed,
            "revisited": revisited
        }
        
        self.interactions.append(interaction)
        self.content_engagement[content_id].append(interaction)
    
    def calculate_engagement_score(
        self,
        student_id: str,
        days: int = 7
    ) -> Dict[str, Any]:
        """
        Calculate overall engagement score
        
        Args:
            student_id: Student identifier
            days: Time period to analyze
        
        Returns:
            Engagement metrics and score
        """
        
        cutoff = datetime.utcnow() - timedelta(days=days)
        
        # Filter student's recent data
        recent_sessions = [
            s for s in self.sessions
            if s["student_id"] == student_id and s["timestamp"] > cutoff
        ]
        
        recent_interactions = [
            i for i in self.interactions
            if i["student_id"] == student_id and i["timestamp"] > cutoff
        ]
        
        if not recent_sessions and not recent_interactions:
            return {
                "engagement_score": 0.0,
                "level": "none",
                "message": f"No activity in last {days} days"
            }
        
        # Calculate component scores
        frequency_score = self._calculate_frequency_score(recent_sessions, days)
        duration_score = self._calculate_duration_score(recent_sessions)
        completion_score = self._calculate_completion_score(recent_sessions)
        interaction_score = self._calculate_interaction_score(recent_interactions)
        voluntary_score = self._calculate_voluntary_score(recent_sessions)
        
        # Weighted average
        engagement_score = (
            frequency_score * 0.25 +
            duration_score * 0.20 +
            completion_score * 0.25 +
            interaction_score * 0.20 +
            voluntary_score * 0.10
        )
        
        # Determine engagement level
        if engagement_score >= 0.8:
            level = "highly_engaged"
        elif engagement_score >= 0.6:
            level = "engaged"
        elif engagement_score >= 0.4:
            level = "moderately_engaged"
        elif engagement_score >= 0.2:
            level = "low_engagement"
        else:
            level = "disengaged"
        
        return {
            "engagement_score": round(engagement_score, 3),
            "level": level,
            "period_days": days,
            "component_scores": {
                "frequency": round(frequency_score, 3),
                "duration": round(duration_score, 3),
                "completion": round(completion_score, 3),
                "interaction": round(interaction_score, 3),
                "voluntary": round(voluntary_score, 3)
            },
            "sessions": len(recent_sessions),
            "total_time": sum(s["duration"] for s in recent_sessions),
            "avg_session_length": sum(s["duration"] for s in recent_sessions) / len(recent_sessions) if recent_sessions else 0
        }
    
    def _calculate_frequency_score(
        self,
        sessions: List[Dict[str, Any]],
        days: int
    ) -> float:
        """Calculate frequency score (0.0 to 1.0)"""
        
        if not sessions:
            return 0.0
        
        # Count unique days with activity
        dates = set(s["timestamp"].date() for s in sessions)
        active_days = len(dates)
        
        # Ideal: Activity every day
        frequency = active_days / days
        
        return min(1.0, frequency)
    
    def _calculate_duration_score(
        self,
        sessions: List[Dict[str, Any]]
    ) -> float:
        """Calculate duration score based on time spent"""
        
        if not sessions:
            return 0.0
        
        total_time = sum(s["duration"] for s in sessions)
        
        # Target: 30+ minutes per session on average
        avg_duration = total_time / len(sessions)
        
        # Score based on average duration
        if avg_duration >= 30:
            return 1.0
        elif avg_duration >= 20:
            return 0.8
        elif avg_duration >= 10:
            return 0.5
        else:
            return 0.3
    
    def _calculate_completion_score(
        self,
        sessions: List[Dict[str, Any]]
    ) -> float:
        """Calculate completion score"""
        
        if not sessions:
            return 0.0
        
        avg_completion = sum(s["completion_rate"] for s in sessions) / len(sessions)
        
        return avg_completion
    
    def _calculate_interaction_score(
        self,
        interactions: List[Dict[str, Any]]
    ) -> float:
        """Calculate interaction quality score"""
        
        if not interactions:
            return 0.0
        
        # Factors:
        # - Completion rate
        # - Revisit rate (shows deeper engagement)
        # - Variety of interactions
        
        completed = sum(1 for i in interactions if i["completed"])
        completion_rate = completed / len(interactions)
        
        revisited = sum(1 for i in interactions if i["revisited"])
        revisit_rate = revisited / len(interactions)
        
        # Count unique content
        unique_content = len(set(i["content_id"] for i in interactions))
        variety_score = min(1.0, unique_content / 10)  # Normalize
        
        score = (
            completion_rate * 0.5 +
            revisit_rate * 0.3 +
            variety_score * 0.2
        )
        
        return score
    
    def _calculate_voluntary_score(
        self,
        sessions: List[Dict[str, Any]]
    ) -> float:
        """Calculate voluntary engagement score"""
        
        if not sessions:
            return 0.0
        
        voluntary_sessions = sum(1 for s in sessions if s.get("voluntary", True))
        
        return voluntary_sessions / len(sessions)
    
    def get_engagement_trends(
        self,
        student_id: str,
        weeks: int = 4
    ) -> Dict[str, Any]:
        """
        Get engagement trends over time
        
        Args:
            student_id: Student identifier
            weeks: Number of weeks to analyze
        
        Returns:
            Trend analysis
        """
        
        cutoff = datetime.utcnow() - timedelta(weeks=weeks)
        
        student_sessions = [
            s for s in self.sessions
            if s["student_id"] == student_id and s["timestamp"] > cutoff
        ]
        
        if not student_sessions:
            return {"trend": "no_data"}
        
        # Group by week
        weekly_scores = []
        
        for week in range(weeks):
            week_start = datetime.utcnow() - timedelta(weeks=weeks-week)
            week_end = week_start + timedelta(weeks=1)
            
            week_sessions = [
                s for s in student_sessions
                if week_start <= s["timestamp"] < week_end
            ]
            
            if week_sessions:
                # Calculate simplified engagement for week
                score = sum(s["completion_rate"] for s in week_sessions) / len(week_sessions)
                weekly_scores.append(score)
            else:
                weekly_scores.append(0.0)
        
        # Analyze trend
        if len(weekly_scores) < 2:
            trend = "insufficient_data"
        else:
            recent_avg = sum(weekly_scores[-2:]) / 2
            earlier_avg = sum(weekly_scores[:2]) / 2
            
            if recent_avg > earlier_avg + 0.15:
                trend = "improving"
            elif recent_avg < earlier_avg - 0.15:
                trend = "declining"
            else:
                trend = "stable"
        
        return {
            "trend": trend,
            "weekly_scores": [round(s, 3) for s in weekly_scores],
            "current_week": round(weekly_scores[-1], 3) if weekly_scores else 0,
            "average": round(sum(weekly_scores) / len(weekly_scores), 3) if weekly_scores else 0
        }
    
    def identify_at_risk_students(
        self,
        threshold: float = 0.4,
        days: int = 14
    ) -> List[Dict[str, Any]]:
        """
        Identify students with low engagement
        
        Args:
            threshold: Engagement score threshold
            days: Period to check
        
        Returns:
            List of at-risk students
        """
        
        # Get unique student IDs
        student_ids = set(s["student_id"] for s in self.sessions)
        
        at_risk = []
        
        for student_id in student_ids:
            score_data = self.calculate_engagement_score(student_id, days)
            
            if score_data["engagement_score"] < threshold:
                at_risk.append({
                    "student_id": student_id,
                    "engagement_score": score_data["engagement_score"],
                    "level": score_data["level"],
                    "sessions": score_data["sessions"],
                    "last_activity": self._get_last_activity(student_id)
                })
        
        # Sort by score (lowest first)
        at_risk.sort(key=lambda x: x["engagement_score"])
        
        return at_risk
    
    def _get_last_activity(self, student_id: str) -> Optional[datetime]:
        """Get student's last activity timestamp"""
        
        student_sessions = [s for s in self.sessions if s["student_id"] == student_id]
        
        if not student_sessions:
            return None
        
        return max(s["timestamp"] for s in student_sessions)