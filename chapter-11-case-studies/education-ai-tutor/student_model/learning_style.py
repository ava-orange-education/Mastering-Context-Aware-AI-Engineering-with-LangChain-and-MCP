"""
Learning Style Analyzer

Identifies and adapts to student's learning style preferences
"""

from typing import Dict, Any, List, Optional
import logging
from datetime import datetime
from collections import defaultdict

logger = logging.getLogger(__name__)


class LearningStyleAnalyzer:
    """
    Analyzes and tracks student's learning style preferences
    """
    
    def __init__(self, student_id: str):
        self.student_id = student_id
        
        # Learning style preferences (0.0 to 1.0)
        self.style_preferences = {
            "visual": 0.33,      # Learns through seeing (diagrams, charts, images)
            "auditory": 0.33,    # Learns through hearing (explanations, discussions)
            "kinesthetic": 0.33  # Learns through doing (hands-on, interactive)
        }
        
        # Content type effectiveness tracking
        self.content_effectiveness: Dict[str, List[float]] = defaultdict(list)
        
        # Engagement metrics by style
        self.engagement_by_style: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        
        # Pace preference
        self.pace_preference = "moderate"  # slow, moderate, fast
        
        # Preferred explanation depth
        self.explanation_depth = "moderate"  # brief, moderate, detailed
        
        # Social learning preference
        self.prefers_collaborative = False
        
        # Updated timestamp
        self.updated_at = datetime.utcnow()
    
    def record_interaction(
        self,
        content_type: str,
        learning_style: str,
        effectiveness: float,
        engagement_metrics: Dict[str, Any]
    ) -> None:
        """
        Record interaction to refine learning style understanding
        
        Args:
            content_type: Type of content (diagram, explanation, exercise, etc.)
            learning_style: Primary style used (visual, auditory, kinesthetic)
            effectiveness: How effective it was (0.0 to 1.0)
            engagement_metrics: {
                "time_spent": float,
                "completion": bool,
                "revisited": bool
            }
        """
        
        # Record effectiveness
        self.content_effectiveness[content_type].append(effectiveness)
        
        # Record engagement
        self.engagement_by_style[learning_style].append({
            "content_type": content_type,
            "effectiveness": effectiveness,
            **engagement_metrics,
            "timestamp": datetime.utcnow()
        })
        
        # Update style preferences
        self._update_style_preferences()
        
        self.updated_at = datetime.utcnow()
        
        logger.info(
            f"Recorded {learning_style} interaction for {self.student_id}: "
            f"{effectiveness:.2f} effectiveness"
        )
    
    def _update_style_preferences(self) -> None:
        """Update learning style preferences based on interaction history"""
        
        # Calculate effectiveness by style
        style_effectiveness = {}
        
        for style in ["visual", "auditory", "kinesthetic"]:
            interactions = self.engagement_by_style.get(style, [])
            
            if not interactions:
                style_effectiveness[style] = 0.33  # Default
                continue
            
            # Weight recent interactions more
            total_weight = 0.0
            weighted_sum = 0.0
            
            for i, interaction in enumerate(interactions[-20:]):  # Last 20
                recency_weight = 0.5 ** ((len(interactions) - i - 1) / 10)
                effectiveness = interaction.get("effectiveness", 0.5)
                
                weighted_sum += effectiveness * recency_weight
                total_weight += recency_weight
            
            style_effectiveness[style] = weighted_sum / total_weight if total_weight > 0 else 0.33
        
        # Normalize to sum to 1.0
        total = sum(style_effectiveness.values())
        if total > 0:
            self.style_preferences = {
                style: score / total
                for style, score in style_effectiveness.items()
            }
    
    def get_dominant_style(self) -> str:
        """Get the student's dominant learning style"""
        
        return max(self.style_preferences.items(), key=lambda x: x[1])[0]
    
    def get_style_distribution(self) -> Dict[str, float]:
        """Get learning style distribution"""
        
        return {
            style: round(score, 3)
            for style, score in self.style_preferences.items()
        }
    
    def recommend_content_types(self, top_n: int = 3) -> List[str]:
        """
        Recommend most effective content types for this student
        
        Args:
            top_n: Number of recommendations
        
        Returns:
            List of content types
        """
        
        # Calculate average effectiveness by content type
        type_scores = []
        
        for content_type, effectiveness_scores in self.content_effectiveness.items():
            if effectiveness_scores:
                avg_effectiveness = sum(effectiveness_scores) / len(effectiveness_scores)
                type_scores.append((content_type, avg_effectiveness))
        
        # Sort by effectiveness
        type_scores.sort(key=lambda x: x[1], reverse=True)
        
        return [content_type for content_type, _ in type_scores[:top_n]]
    
    def get_personalization_recommendations(self) -> Dict[str, Any]:
        """Get recommendations for personalizing learning experience"""
        
        dominant_style = self.get_dominant_style()
        
        recommendations = {
            "dominant_style": dominant_style,
            "style_distribution": self.get_style_distribution(),
            "pace": self.pace_preference,
            "explanation_depth": self.explanation_depth
        }
        
        # Style-specific recommendations
        if dominant_style == "visual":
            recommendations["content_suggestions"] = [
                "Use diagrams and charts frequently",
                "Color-code important information",
                "Provide visual step-by-step guides",
                "Include infographics and flowcharts"
            ]
        elif dominant_style == "auditory":
            recommendations["content_suggestions"] = [
                "Provide detailed verbal explanations",
                "Use analogies and stories",
                "Include discussion prompts",
                "Suggest read-aloud of key concepts"
            ]
        else:  # kinesthetic
            recommendations["content_suggestions"] = [
                "Include interactive exercises",
                "Provide hands-on activities",
                "Use real-world applications",
                "Incorporate physical analogies"
            ]
        
        # Recommended content types
        recommendations["preferred_content_types"] = self.recommend_content_types(3)
        
        return recommendations
    
    def set_pace_preference(self, pace: str) -> None:
        """
        Set student's pace preference
        
        Args:
            pace: slow, moderate, or fast
        """
        
        if pace not in ["slow", "moderate", "fast"]:
            raise ValueError("Pace must be: slow, moderate, or fast")
        
        self.pace_preference = pace
        self.updated_at = datetime.utcnow()
    
    def set_explanation_depth(self, depth: str) -> None:
        """
        Set preferred explanation depth
        
        Args:
            depth: brief, moderate, or detailed
        """
        
        if depth not in ["brief", "moderate", "detailed"]:
            raise ValueError("Depth must be: brief, moderate, or detailed")
        
        self.explanation_depth = depth
        self.updated_at = datetime.utcnow()
    
    def analyze_engagement_patterns(self) -> Dict[str, Any]:
        """Analyze engagement patterns across learning styles"""
        
        patterns = {}
        
        for style, interactions in self.engagement_by_style.items():
            if not interactions:
                continue
            
            # Calculate metrics
            avg_time = sum(i.get("time_spent", 0) for i in interactions) / len(interactions)
            completion_rate = sum(1 for i in interactions if i.get("completion")) / len(interactions)
            revisit_rate = sum(1 for i in interactions if i.get("revisited")) / len(interactions)
            avg_effectiveness = sum(i.get("effectiveness", 0) for i in interactions) / len(interactions)
            
            patterns[style] = {
                "avg_time_spent": round(avg_time, 1),
                "completion_rate": round(completion_rate, 3),
                "revisit_rate": round(revisit_rate, 3),
                "avg_effectiveness": round(avg_effectiveness, 3),
                "total_interactions": len(interactions)
            }
        
        return patterns