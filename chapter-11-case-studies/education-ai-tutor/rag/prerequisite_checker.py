"""
Prerequisite Checker

Checks if student has necessary prerequisite knowledge
"""

from typing import Dict, Any, List, Set, Optional
import logging

logger = logging.getLogger(__name__)


class PrerequisiteChecker:
    """
    Checks and manages prerequisite knowledge requirements
    """
    
    def __init__(self):
        # Prerequisite graph
        # In production, load from curriculum database
        self.prerequisites: Dict[str, List[str]] = {}
        
        # Minimum mastery level required for prerequisites
        self.min_prerequisite_mastery = 0.7
    
    def set_prerequisites(
        self,
        concept: str,
        prerequisites: List[str]
    ) -> None:
        """
        Set prerequisites for a concept
        
        Args:
            concept: Concept name
            prerequisites: List of prerequisite concepts
        """
        self.prerequisites[concept] = prerequisites
    
    def check_readiness(
        self,
        concept: str,
        student_mastery: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Check if student is ready to learn a concept
        
        Args:
            concept: Concept to check
            student_mastery: Dict mapping concepts to mastery levels
        
        Returns:
            Readiness assessment
        """
        
        prerequisites = self.prerequisites.get(concept, [])
        
        if not prerequisites:
            return {
                "ready": True,
                "missing_prerequisites": [],
                "weak_prerequisites": [],
                "message": f"No prerequisites for {concept}"
            }
        
        missing = []
        weak = []
        
        for prereq in prerequisites:
            mastery = student_mastery.get(prereq, 0.0)
            
            if mastery == 0.0:
                missing.append(prereq)
            elif mastery < self.min_prerequisite_mastery:
                weak.append({
                    "concept": prereq,
                    "current_mastery": mastery,
                    "required_mastery": self.min_prerequisite_mastery
                })
        
        ready = len(missing) == 0 and len(weak) == 0
        
        return {
            "ready": ready,
            "missing_prerequisites": missing,
            "weak_prerequisites": weak,
            "total_prerequisites": len(prerequisites),
            "message": self._generate_readiness_message(concept, ready, missing, weak)
        }
    
    def _generate_readiness_message(
        self,
        concept: str,
        ready: bool,
        missing: List[str],
        weak: List[Dict[str, Any]]
    ) -> str:
        """Generate human-readable readiness message"""
        
        if ready:
            return f"Ready to learn {concept}!"
        
        parts = [f"Not quite ready for {concept}."]
        
        if missing:
            parts.append(f"Need to learn: {', '.join(missing)}")
        
        if weak:
            weak_concepts = [w["concept"] for w in weak]
            parts.append(f"Need more practice with: {', '.join(weak_concepts)}")
        
        return " ".join(parts)
    
    def get_learning_path(
        self,
        target_concept: str,
        student_mastery: Dict[str, float]
    ) -> List[str]:
        """
        Get recommended learning path to reach target concept
        
        Args:
            target_concept: Goal concept
            student_mastery: Current mastery levels
        
        Returns:
            Ordered list of concepts to learn
        """
        
        # Topological sort of prerequisite graph
        path = []
        visited = set()
        
        def visit(concept: str):
            if concept in visited:
                return
            
            visited.add(concept)
            
            # Visit prerequisites first
            for prereq in self.prerequisites.get(concept, []):
                # Only include if not already mastered
                if student_mastery.get(prereq, 0.0) < self.min_prerequisite_mastery:
                    visit(prereq)
            
            # Add this concept if not mastered
            if student_mastery.get(concept, 0.0) < 0.8:
                path.append(concept)
        
        visit(target_concept)
        
        return path
    
    def get_next_concepts(
        self,
        student_mastery: Dict[str, float],
        max_concepts: int = 5
    ) -> List[str]:
        """
        Get recommended next concepts to learn
        
        Args:
            student_mastery: Current mastery levels
            max_concepts: Maximum concepts to return
        
        Returns:
            List of recommended concepts
        """
        
        candidates = []
        
        for concept, prereqs in self.prerequisites.items():
            # Skip if already mastered
            if student_mastery.get(concept, 0.0) >= 0.8:
                continue
            
            # Check if prerequisites are met
            readiness = self.check_readiness(concept, student_mastery)
            
            if readiness["ready"]:
                candidates.append(concept)
            elif not readiness["missing_prerequisites"] and readiness["weak_prerequisites"]:
                # Close to ready - good candidate
                candidates.insert(0, concept)
        
        return candidates[:max_concepts]
    
    def visualize_prerequisite_tree(
        self,
        concept: str,
        student_mastery: Dict[str, float],
        depth: int = 0,
        max_depth: int = 3
    ) -> str:
        """
        Create a text visualization of prerequisite tree
        
        Args:
            concept: Root concept
            student_mastery: Student's mastery levels
            depth: Current depth (for recursion)
            max_depth: Maximum depth to show
        
        Returns:
            Tree visualization string
        """
        
        if depth >= max_depth:
            return ""
        
        indent = "  " * depth
        mastery = student_mastery.get(concept, 0.0)
        
        # Status indicator
        if mastery >= 0.8:
            status = "✓"
        elif mastery >= 0.5:
            status = "◐"
        else:
            status = "○"
        
        lines = [f"{indent}{status} {concept} ({mastery:.0%})"]
        
        # Add prerequisites
        prereqs = self.prerequisites.get(concept, [])
        for prereq in prereqs:
            subtree = self.visualize_prerequisite_tree(
                prereq,
                student_mastery,
                depth + 1,
                max_depth
            )
            if subtree:
                lines.append(subtree)
        
        return "\n".join(lines)
    
    def build_prerequisite_graph_from_curriculum(
        self,
        curriculum: Dict[str, Any]
    ) -> None:
        """
        Build prerequisite graph from curriculum specification
        
        Args:
            curriculum: Curriculum with prerequisite information
        """
        
        # Example curriculum format:
        # {
        #   "algebra_basics": {
        #     "prerequisites": ["arithmetic"],
        #     ...
        #   }
        # }
        
        for concept, details in curriculum.items():
            if "prerequisites" in details:
                self.set_prerequisites(concept, details["prerequisites"])
        
        logger.info(f"Built prerequisite graph with {len(self.prerequisites)} concepts")