"""
Content Recommender

Recommends personalized content based on student profile and progress
"""

from typing import Dict, Any, List, Optional
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class ContentRecommender:
    """
    Recommends personalized learning content
    """
    
    def __init__(self):
        # Content effectiveness tracking
        self.content_effectiveness: Dict[str, List[float]] = {}
        
        # Student preferences
        self.student_preferences: Dict[str, Dict[str, Any]] = {}
    
    def recommend_content(
        self,
        student_id: str,
        topic: str,
        learning_style: str,
        mastery_level: float,
        interests: List[str],
        previous_content: List[str],
        max_recommendations: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Recommend content for a student
        
        Args:
            student_id: Student identifier
            topic: Topic to learn
            learning_style: Student's learning style
            mastery_level: Current mastery of topic
            interests: Student's interests
            previous_content: Previously seen content IDs
            max_recommendations: Maximum recommendations
        
        Returns:
            List of content recommendations
        """
        
        logger.info(f"Generating content recommendations for {student_id} on {topic}")
        
        # Generate candidate content
        candidates = self._generate_candidates(
            topic=topic,
            learning_style=learning_style,
            mastery_level=mastery_level,
            interests=interests
        )
        
        # Filter out previously seen content
        candidates = [c for c in candidates if c["id"] not in previous_content]
        
        # Score and rank candidates
        scored_candidates = self._score_candidates(
            candidates=candidates,
            student_id=student_id,
            learning_style=learning_style,
            mastery_level=mastery_level
        )
        
        # Sort by score
        scored_candidates.sort(key=lambda x: x["score"], reverse=True)
        
        # Return top recommendations
        recommendations = scored_candidates[:max_recommendations]
        
        logger.info(f"Generated {len(recommendations)} recommendations")
        
        return recommendations
    
    def _generate_candidates(
        self,
        topic: str,
        learning_style: str,
        mastery_level: float,
        interests: List[str]
    ) -> List[Dict[str, Any]]:
        """Generate candidate content"""
        
        candidates = []
        
        # Different content types based on learning style
        if learning_style == "visual":
            content_types = ["diagram", "video", "infographic", "interactive_viz"]
        elif learning_style == "auditory":
            content_types = ["explanation", "audio", "discussion", "lecture"]
        elif learning_style == "kinesthetic":
            content_types = ["hands_on", "simulation", "interactive", "project"]
        else:
            content_types = ["explanation", "example", "exercise", "video"]
        
        # Generate content for each type
        for content_type in content_types:
            candidates.append({
                "id": f"{topic}_{content_type}_{len(candidates)}",
                "topic": topic,
                "type": content_type,
                "learning_style": learning_style,
                "difficulty": self._map_mastery_to_difficulty(mastery_level),
                "estimated_time": self._estimate_time(content_type),
                "has_interest_connection": any(interest in topic.lower() for interest in interests)
            })
        
        # Add variety - different difficulty levels
        for difficulty in ["easy", "medium", "hard"]:
            candidates.append({
                "id": f"{topic}_exercise_{difficulty}_{len(candidates)}",
                "topic": topic,
                "type": "exercise",
                "learning_style": "balanced",
                "difficulty": difficulty,
                "estimated_time": 15,
                "has_interest_connection": False
            })
        
        return candidates
    
    def _score_candidates(
        self,
        candidates: List[Dict[str, Any]],
        student_id: str,
        learning_style: str,
        mastery_level: float
    ) -> List[Dict[str, Any]]:
        """Score candidate content"""
        
        for candidate in candidates:
            score = 0.0
            
            # Learning style match (0-30 points)
            if candidate["learning_style"] == learning_style:
                score += 30
            elif candidate["learning_style"] == "balanced":
                score += 15
            
            # Difficulty appropriateness (0-25 points)
            target_difficulty = self._map_mastery_to_difficulty(mastery_level)
            if candidate["difficulty"] == target_difficulty:
                score += 25
            elif abs(self._difficulty_to_num(candidate["difficulty"]) - 
                    self._difficulty_to_num(target_difficulty)) == 1:
                score += 15
            
            # Interest connection (0-20 points)
            if candidate.get("has_interest_connection"):
                score += 20
            
            # Content type variety (0-15 points)
            # Prefer types that student hasn't seen recently
            student_prefs = self.student_preferences.get(student_id, {})
            recent_types = student_prefs.get("recent_content_types", [])
            if candidate["type"] not in recent_types[-3:]:
                score += 15
            
            # Historical effectiveness (0-10 points)
            effectiveness = self._get_content_effectiveness(
                student_id,
                candidate["type"]
            )
            score += effectiveness * 10
            
            candidate["score"] = score
            candidate["score_breakdown"] = {
                "learning_style_match": 30 if candidate["learning_style"] == learning_style else 0,
                "difficulty_match": 25 if candidate["difficulty"] == target_difficulty else 0,
                "interest_connection": 20 if candidate.get("has_interest_connection") else 0,
                "variety": 15 if candidate["type"] not in recent_types[-3:] else 0,
                "effectiveness": round(effectiveness * 10, 1)
            }
        
        return candidates
    
    def _map_mastery_to_difficulty(self, mastery: float) -> str:
        """Map mastery level to difficulty"""
        if mastery < 0.3:
            return "beginner"
        elif mastery < 0.5:
            return "easy"
        elif mastery < 0.7:
            return "medium"
        elif mastery < 0.85:
            return "hard"
        else:
            return "advanced"
    
    def _difficulty_to_num(self, difficulty: str) -> int:
        """Convert difficulty to number for comparison"""
        levels = ["beginner", "easy", "medium", "hard", "advanced", "expert"]
        try:
            return levels.index(difficulty)
        except ValueError:
            return 2  # Default to medium
    
    def _estimate_time(self, content_type: str) -> int:
        """Estimate time needed for content type (minutes)"""
        time_estimates = {
            "explanation": 10,
            "example": 5,
            "exercise": 15,
            "video": 12,
            "diagram": 8,
            "interactive": 20,
            "hands_on": 30,
            "project": 60
        }
        return time_estimates.get(content_type, 15)
    
    def _get_content_effectiveness(
        self,
        student_id: str,
        content_type: str
    ) -> float:
        """Get historical effectiveness of content type for student"""
        
        key = f"{student_id}_{content_type}"
        effectiveness_scores = self.content_effectiveness.get(key, [])
        
        if not effectiveness_scores:
            return 0.5  # Default
        
        # Weight recent scores more heavily
        if len(effectiveness_scores) <= 3:
            return sum(effectiveness_scores) / len(effectiveness_scores)
        
        recent = effectiveness_scores[-3:]
        return sum(recent) / len(recent)
    
    def record_content_effectiveness(
        self,
        student_id: str,
        content_type: str,
        effectiveness: float
    ) -> None:
        """
        Record how effective content was
        
        Args:
            student_id: Student identifier
            content_type: Type of content
            effectiveness: Effectiveness score (0.0 to 1.0)
        """
        
        key = f"{student_id}_{content_type}"
        
        if key not in self.content_effectiveness:
            self.content_effectiveness[key] = []
        
        self.content_effectiveness[key].append(effectiveness)
        
        # Keep only recent scores (last 20)
        self.content_effectiveness[key] = self.content_effectiveness[key][-20:]
        
        logger.info(f"Recorded {content_type} effectiveness for {student_id}: {effectiveness:.2f}")
    
    def update_student_preferences(
        self,
        student_id: str,
        content_consumed: Dict[str, Any]
    ) -> None:
        """
        Update student's content preferences
        
        Args:
            student_id: Student identifier
            content_consumed: Content that was consumed
        """
        
        if student_id not in self.student_preferences:
            self.student_preferences[student_id] = {
                "recent_content_types": [],
                "preferred_types": [],
                "last_updated": datetime.utcnow()
            }
        
        prefs = self.student_preferences[student_id]
        
        # Add to recent types
        content_type = content_consumed.get("type")
        if content_type:
            prefs["recent_content_types"].append(content_type)
            prefs["recent_content_types"] = prefs["recent_content_types"][-10:]
        
        prefs["last_updated"] = datetime.utcnow()