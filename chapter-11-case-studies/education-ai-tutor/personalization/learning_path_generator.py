"""
Learning Path Generator

Generates personalized learning paths based on goals and current knowledge
"""

from typing import Dict, Any, List, Optional, Set
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class LearningPathGenerator:
    """
    Generates personalized learning paths
    """
    
    def __init__(self):
        # Concept dependencies
        self.concept_graph: Dict[str, List[str]] = {}
        
        # Estimated time per concept
        self.concept_times: Dict[str, int] = {}
        
        # Concept difficulty
        self.concept_difficulty: Dict[str, str] = {}
    
    def generate_learning_path(
        self,
        student_id: str,
        learning_goal: str,
        current_knowledge: Dict[str, float],
        target_mastery: float = 0.8,
        time_budget: Optional[int] = None,
        deadline: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Generate a personalized learning path
        
        Args:
            student_id: Student identifier
            learning_goal: Target concept to master
            current_knowledge: Current mastery levels
            target_mastery: Desired mastery level
            time_budget: Available time in minutes
            deadline: Target completion date
        
        Returns:
            Learning path with milestones
        """
        
        logger.info(f"Generating learning path for {student_id} to learn {learning_goal}")
        
        # Find required concepts
        required_concepts = self._find_required_concepts(
            target=learning_goal,
            current_knowledge=current_knowledge,
            target_mastery=target_mastery
        )
        
        # Order concepts (topological sort)
        ordered_path = self._order_concepts(
            concepts=required_concepts,
            current_knowledge=current_knowledge
        )
        
        # Estimate time required
        total_time = self._estimate_total_time(ordered_path, current_knowledge)
        
        # Create milestones
        milestones = self._create_milestones(
            path=ordered_path,
            deadline=deadline,
            time_budget=time_budget
        )
        
        # Generate recommendations
        recommendations = self._generate_path_recommendations(
            path=ordered_path,
            time_budget=time_budget,
            total_time=total_time,
            deadline=deadline
        )
        
        return {
            "learning_goal": learning_goal,
            "path": ordered_path,
            "milestones": milestones,
            "estimated_time_minutes": total_time,
            "estimated_days": self._estimate_days(total_time),
            "concepts_to_learn": len(ordered_path),
            "recommendations": recommendations,
            "feasibility": self._assess_feasibility(
                total_time=total_time,
                time_budget=time_budget,
                deadline=deadline
            )
        }
    
    def _find_required_concepts(
        self,
        target: str,
        current_knowledge: Dict[str, float],
        target_mastery: float
    ) -> Set[str]:
        """Find all concepts needed to reach target"""
        
        required = set()
        
        def traverse(concept: str):
            # Skip if already mastered
            if current_knowledge.get(concept, 0.0) >= target_mastery:
                return
            
            required.add(concept)
            
            # Add prerequisites
            prerequisites = self.concept_graph.get(concept, [])
            for prereq in prerequisites:
                traverse(prereq)
        
        traverse(target)
        
        return required
    
    def _order_concepts(
        self,
        concepts: Set[str],
        current_knowledge: Dict[str, float]
    ) -> List[Dict[str, Any]]:
        """Order concepts using topological sort"""
        
        # Create adjacency list for required concepts
        graph = {c: [] for c in concepts}
        in_degree = {c: 0 for c in concepts}
        
        for concept in concepts:
            prereqs = self.concept_graph.get(concept, [])
            for prereq in prereqs:
                if prereq in concepts:
                    graph[prereq].append(concept)
                    in_degree[concept] += 1
        
        # Topological sort
        queue = [c for c in concepts if in_degree[c] == 0]
        ordered = []
        
        while queue:
            # Sort by current knowledge (learn partially known first)
            queue.sort(key=lambda c: current_knowledge.get(c, 0.0), reverse=True)
            
            current = queue.pop(0)
            
            ordered.append({
                "concept": current,
                "current_mastery": current_knowledge.get(current, 0.0),
                "difficulty": self.concept_difficulty.get(current, "medium"),
                "estimated_time": self.concept_times.get(current, 60),
                "prerequisites": self.concept_graph.get(current, [])
            })
            
            # Update in-degrees
            for neighbor in graph[current]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        return ordered
    
    def _estimate_total_time(
        self,
        path: List[Dict[str, Any]],
        current_knowledge: Dict[str, float]
    ) -> int:
        """Estimate total time needed in minutes"""
        
        total = 0
        
        for step in path:
            base_time = step["estimated_time"]
            current_mastery = step["current_mastery"]
            
            # Reduce time if partially learned
            if current_mastery > 0:
                time_reduction = current_mastery * 0.6
                adjusted_time = base_time * (1 - time_reduction)
            else:
                adjusted_time = base_time
            
            total += adjusted_time
        
        return int(total)
    
    def _create_milestones(
        self,
        path: List[Dict[str, Any]],
        deadline: Optional[datetime],
        time_budget: Optional[int]
    ) -> List[Dict[str, Any]]:
        """Create milestones along the learning path"""
        
        if not path:
            return []
        
        milestones = []
        
        # Create milestone every 3-5 concepts
        milestone_frequency = min(5, max(3, len(path) // 4))
        
        current_time = 0
        
        for i, step in enumerate(path):
            current_time += step["estimated_time"]
            
            if (i + 1) % milestone_frequency == 0 or i == len(path) - 1:
                milestone = {
                    "milestone_number": len(milestones) + 1,
                    "concept": step["concept"],
                    "concepts_completed": i + 1,
                    "total_concepts": len(path),
                    "cumulative_time": current_time,
                    "progress_percentage": ((i + 1) / len(path)) * 100
                }
                
                # Add deadline if provided
                if deadline:
                    # Distribute time proportionally
                    time_to_deadline = (deadline - datetime.utcnow()).total_seconds() / 60
                    milestone_time = (current_time / self._estimate_total_time(path, {})) * time_to_deadline
                    milestone["target_date"] = datetime.utcnow() + timedelta(minutes=milestone_time)
                
                milestones.append(milestone)
        
        return milestones
    
    def _estimate_days(self, total_minutes: int, minutes_per_day: int = 60) -> int:
        """Estimate days needed"""
        return int(total_minutes / minutes_per_day) + 1
    
    def _assess_feasibility(
        self,
        total_time: int,
        time_budget: Optional[int],
        deadline: Optional[datetime]
    ) -> Dict[str, Any]:
        """Assess if the path is feasible"""
        
        feasible = True
        issues = []
        
        if time_budget and total_time > time_budget:
            feasible = False
            issues.append(f"Requires {total_time} minutes but only {time_budget} available")
        
        if deadline:
            time_to_deadline = (deadline - datetime.utcnow()).total_seconds() / 60
            if total_time > time_to_deadline:
                feasible = False
                issues.append(f"Requires {total_time} minutes but only {int(time_to_deadline)} until deadline")
        
        return {
            "feasible": feasible,
            "issues": issues,
            "utilization": (total_time / time_budget * 100) if time_budget else None
        }
    
    def _generate_path_recommendations(
        self,
        path: List[Dict[str, Any]],
        time_budget: Optional[int],
        total_time: int,
        deadline: Optional[datetime]
    ) -> List[str]:
        """Generate recommendations for following the path"""
        
        recommendations = []
        
        # Pacing recommendation
        if deadline:
            days_available = (deadline - datetime.utcnow()).days
            time_per_day = total_time / days_available if days_available > 0 else total_time
            
            if time_per_day < 30:
                recommendations.append(f"Pace: Study ~{int(time_per_day)} minutes per day")
            elif time_per_day < 120:
                recommendations.append(f"Pace: Study ~{int(time_per_day/60)} hour per day")
            else:
                recommendations.append(f"Pace: Study ~{int(time_per_day/60)} hours per day (intensive)")
        
        # Study strategy recommendations
        if len(path) > 10:
            recommendations.append("Break the path into smaller chunks - focus on one milestone at a time")
        
        if any(step["difficulty"] == "hard" or step["difficulty"] == "advanced" for step in path):
            recommendations.append("Some challenging concepts ahead - allocate extra time for difficult topics")
        
        # Review recommendations
        recommendations.append("Review earlier concepts periodically to maintain retention")
        
        return recommendations
    
    def add_concept_dependency(
        self,
        concept: str,
        prerequisites: List[str],
        estimated_time: int = 60,
        difficulty: str = "medium"
    ) -> None:
        """
        Add a concept to the graph
        
        Args:
            concept: Concept name
            prerequisites: List of prerequisite concepts
            estimated_time: Estimated learning time in minutes
            difficulty: Difficulty level
        """
        
        self.concept_graph[concept] = prerequisites
        self.concept_times[concept] = estimated_time
        self.concept_difficulty[concept] = difficulty