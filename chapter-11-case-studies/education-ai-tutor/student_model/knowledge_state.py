"""
Knowledge State

Tracks student's knowledge and mastery of concepts
"""

from typing import Dict, Any, List, Optional, Set
from datetime import datetime, timedelta
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


class KnowledgeState:
    """
    Represents a student's current knowledge state
    """
    
    def __init__(self, student_id: str):
        self.student_id = student_id
        
        # Concept mastery levels (0.0 to 1.0)
        self.mastery_levels: Dict[str, float] = {}
        
        # Concept dependencies
        self.prerequisites: Dict[str, List[str]] = {}
        
        # Recent performance by concept
        self.recent_performance: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        
        # Last interaction timestamp per concept
        self.last_practiced: Dict[str, datetime] = {}
        
        # Strengths and weaknesses
        self.strengths: Set[str] = set()
        self.weaknesses: Set[str] = set()
        
        # Learning goals
        self.learning_goals: List[Dict[str, Any]] = []
        
        # Metadata
        self.created_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()
    
    def update_mastery(
        self,
        concept: str,
        performance: Dict[str, Any]
    ) -> None:
        """
        Update mastery level for a concept based on performance
        
        Args:
            concept: Concept name
            performance: {
                "correct": bool,
                "difficulty": str,
                "time_taken": float,
                "attempts": int
            }
        """
        
        # Record performance
        self.recent_performance[concept].append({
            **performance,
            "timestamp": datetime.utcnow()
        })
        
        # Keep only recent performances (last 20)
        self.recent_performance[concept] = self.recent_performance[concept][-20:]
        
        # Calculate new mastery level
        new_mastery = self._calculate_mastery(concept)
        self.mastery_levels[concept] = new_mastery
        
        # Update last practiced
        self.last_practiced[concept] = datetime.utcnow()
        
        # Update strengths/weaknesses
        self._update_strengths_weaknesses()
        
        # Update timestamp
        self.updated_at = datetime.utcnow()
        
        logger.info(
            f"Updated mastery for {concept}: {new_mastery:.2%} "
            f"(student: {self.student_id})"
        )
    
    def _calculate_mastery(self, concept: str) -> float:
        """
        Calculate mastery level based on recent performance
        
        Uses weighted average with recency bias and difficulty weighting
        """
        
        performances = self.recent_performance.get(concept, [])
        
        if not performances:
            return 0.0
        
        # Weight recent performances more heavily
        total_weight = 0.0
        weighted_sum = 0.0
        
        # Difficulty multipliers
        difficulty_multipliers = {
            "beginner": 0.6,
            "easy": 0.7,
            "medium": 1.0,
            "hard": 1.3,
            "advanced": 1.5,
            "expert": 1.8
        }
        
        for i, perf in enumerate(performances):
            # Recency weight (exponential decay)
            recency_weight = 0.5 ** ((len(performances) - i - 1) / 5)
            
            # Difficulty weight
            difficulty = perf.get("difficulty", "medium")
            difficulty_weight = difficulty_multipliers.get(difficulty, 1.0)
            
            # Performance score
            if perf.get("correct"):
                # Bonus for fewer attempts
                attempts = perf.get("attempts", 1)
                score = 1.0 / attempts if attempts > 0 else 1.0
            else:
                score = 0.0
            
            # Combined weight
            weight = recency_weight * difficulty_weight
            
            weighted_sum += score * weight
            total_weight += weight
        
        if total_weight == 0:
            return 0.0
        
        mastery = weighted_sum / total_weight
        
        # Apply retention decay if not practiced recently
        mastery = self._apply_retention_decay(concept, mastery)
        
        return min(1.0, max(0.0, mastery))
    
    def _apply_retention_decay(self, concept: str, mastery: float) -> float:
        """Apply retention decay based on time since last practice"""
        
        if concept not in self.last_practiced:
            return mastery
        
        days_since_practice = (datetime.utcnow() - self.last_practiced[concept]).days
        
        # Exponential decay (half-life of 30 days)
        decay_factor = 0.5 ** (days_since_practice / 30)
        
        # Apply decay but maintain some baseline
        decayed_mastery = mastery * (0.2 + 0.8 * decay_factor)
        
        return decayed_mastery
    
    def _update_strengths_weaknesses(self) -> None:
        """Update identified strengths and weaknesses"""
        
        self.strengths.clear()
        self.weaknesses.clear()
        
        for concept, mastery in self.mastery_levels.items():
            if mastery >= 0.8:
                self.strengths.add(concept)
            elif mastery < 0.5:
                self.weaknesses.add(concept)
    
    def get_mastery(self, concept: str) -> float:
        """Get current mastery level for a concept"""
        
        if concept not in self.mastery_levels:
            return 0.0
        
        # Apply retention decay
        mastery = self.mastery_levels[concept]
        return self._apply_retention_decay(concept, mastery)
    
    def is_ready_for(self, concept: str) -> bool:
        """
        Check if student is ready to learn a concept
        
        Args:
            concept: Concept to check readiness for
        
        Returns:
            True if all prerequisites are mastered
        """
        
        prerequisites = self.prerequisites.get(concept, [])
        
        for prereq in prerequisites:
            if self.get_mastery(prereq) < 0.7:
                return False
        
        return True
    
    def get_next_concepts(self, max_concepts: int = 5) -> List[str]:
        """
        Get recommended next concepts to learn
        
        Args:
            max_concepts: Maximum concepts to return
        
        Returns:
            List of concept names ordered by priority
        """
        
        candidates = []
        
        # Find concepts that are ready to learn
        for concept, prereqs in self.prerequisites.items():
            # Skip if already mastered
            if self.get_mastery(concept) >= 0.8:
                continue
            
            # Check if prerequisites are met
            if not self.is_ready_for(concept):
                continue
            
            # Calculate priority
            # Higher priority for:
            # - Learning goals
            # - Partially learned concepts
            # - Recently struggled concepts
            
            priority = 0.0
            
            # Check if in learning goals
            if any(goal.get("concept") == concept for goal in self.learning_goals):
                priority += 10.0
            
            # Partial knowledge bonus
            current_mastery = self.get_mastery(concept)
            if 0.3 <= current_mastery < 0.7:
                priority += 5.0
            
            # Recent struggle bonus
            recent_perfs = self.recent_performance.get(concept, [])[-3:]
            if recent_perfs and not all(p.get("correct") for p in recent_perfs):
                priority += 3.0
            
            candidates.append((concept, priority))
        
        # Sort by priority
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        return [concept for concept, _ in candidates[:max_concepts]]
    
    def set_prerequisites(
        self,
        concept: str,
        prerequisites: List[str]
    ) -> None:
        """Set prerequisites for a concept"""
        self.prerequisites[concept] = prerequisites
    
    def add_learning_goal(
        self,
        concept: str,
        target_mastery: float = 0.8,
        deadline: Optional[datetime] = None
    ) -> None:
        """Add a learning goal"""
        
        goal = {
            "concept": concept,
            "target_mastery": target_mastery,
            "deadline": deadline,
            "created_at": datetime.utcnow()
        }
        
        self.learning_goals.append(goal)
    
    def get_learning_goals_progress(self) -> List[Dict[str, Any]]:
        """Get progress on learning goals"""
        
        progress = []
        
        for goal in self.learning_goals:
            concept = goal["concept"]
            current_mastery = self.get_mastery(concept)
            target = goal["target_mastery"]
            
            progress.append({
                "concept": concept,
                "target_mastery": target,
                "current_mastery": current_mastery,
                "progress": current_mastery / target if target > 0 else 0,
                "deadline": goal.get("deadline"),
                "achieved": current_mastery >= target
            })
        
        return progress
    
    def get_knowledge_map(self) -> Dict[str, Any]:
        """Get visual representation of knowledge state"""
        
        return {
            "student_id": self.student_id,
            "total_concepts": len(self.mastery_levels),
            "mastered_concepts": len(self.strengths),
            "in_progress": len([c for c, m in self.mastery_levels.items() if 0.5 <= m < 0.8]),
            "needs_practice": len(self.weaknesses),
            "strengths": list(self.strengths),
            "weaknesses": list(self.weaknesses),
            "mastery_levels": {
                concept: round(self.get_mastery(concept), 3)
                for concept in self.mastery_levels.keys()
            },
            "next_recommended": self.get_next_concepts(5)
        }