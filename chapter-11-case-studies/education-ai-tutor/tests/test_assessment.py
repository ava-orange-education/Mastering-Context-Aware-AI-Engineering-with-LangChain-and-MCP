"""
Tests for assessment components
"""

import pytest
from datetime import datetime
import sys
sys.path.append('../..')

from evaluation.learning_outcomes import LearningOutcomesEvaluator
from evaluation.pedagogical_quality import PedagogicalQualityEvaluator


class TestLearningOutcomesEvaluator:
    """Test Learning Outcomes Evaluator"""
    
    def test_initialization(self):
        """Test evaluator initializes"""
        evaluator = LearningOutcomesEvaluator()
        
        assert evaluator.pre_assessments is not None
        assert evaluator.post_assessments is not None
    
    def test_record_pre_assessment(self):
        """Test recording pre-assessment"""
        evaluator = LearningOutcomesEvaluator()
        
        evaluator.record_pre_assessment(
            student_id="test_student",
            topic="algebra",
            assessment_results={"score": 0.5, "mastery_level": 0.45}
        )
        
        key = "test_student_algebra"
        assert key in evaluator.pre_assessments
        assert evaluator.pre_assessments[key]["score"] == 0.5
    
    def test_record_post_assessment(self):
        """Test recording post-assessment"""
        evaluator = LearningOutcomesEvaluator()
        
        evaluator.record_post_assessment(
            student_id="test_student",
            topic="algebra",
            assessment_results={"score": 0.85, "mastery_level": 0.8}
        )
        
        key = "test_student_algebra"
        assert key in evaluator.post_assessments
        assert evaluator.post_assessments[key]["score"] == 0.85
    
    def test_calculate_learning_gain(self):
        """Test learning gain calculation"""
        evaluator = LearningOutcomesEvaluator()
        
        # Record pre and post
        evaluator.record_pre_assessment(
            student_id="test_student",
            topic="algebra",
            assessment_results={"score": 0.4, "mastery_level": 0.35}
        )
        
        evaluator.record_post_assessment(
            student_id="test_student",
            topic="algebra",
            assessment_results={"score": 0.85, "mastery_level": 0.8}
        )
        
        gain = evaluator.calculate_learning_gain("test_student", "algebra")
        
        assert gain["available"] is True
        assert gain["absolute_gain"] > 0
        assert gain["normalized_gain"] > 0
        assert gain["gain_level"] in ["high", "medium", "low", "none"]
    
    def test_no_assessments(self):
        """Test when no assessments exist"""
        evaluator = LearningOutcomesEvaluator()
        
        gain = evaluator.calculate_learning_gain("nonexistent", "topic")
        
        assert gain["available"] is False


class TestPedagogicalQualityEvaluator:
    """Test Pedagogical Quality Evaluator"""
    
    def test_initialization(self):
        """Test evaluator initializes"""
        evaluator = PedagogicalQualityEvaluator()
        
        assert evaluator.llm is not None
    
    def test_evaluate_feedback_quality(self):
        """Test feedback quality evaluation"""
        evaluator = PedagogicalQualityEvaluator()
        
        feedback = "Great job! You correctly identified the main concept. Next time, try to explain your reasoning in more detail."
        student_answer = "Photosynthesis is how plants make food"
        
        quality = evaluator.evaluate_feedback_quality(
            feedback=feedback,
            student_answer=student_answer,
            correctness=True
        )
        
        assert "quality_score" in quality
        assert "metrics" in quality
        assert "passes_quality" in quality
        assert 0 <= quality["quality_score"] <= 1
    
    def test_check_specificity(self):
        """Test specificity check"""
        evaluator = PedagogicalQualityEvaluator()
        
        generic = "Good job!"
        specific = "Your explanation of photosynthesis clearly identifies the role of chlorophyll in capturing light energy."
        
        generic_score = evaluator._check_specificity(generic)
        specific_score = evaluator._check_specificity(specific)
        
        assert specific_score > generic_score
    
    def test_check_constructiveness(self):
        """Test constructiveness check"""
        evaluator = PedagogicalQualityEvaluator()
        
        constructive = "Consider reviewing the relationship between mass and weight. You could try drawing a diagram to visualize the difference."
        non_constructive = "That's wrong."
        
        constructive_score = evaluator._check_constructiveness(constructive)
        non_constructive_score = evaluator._check_constructiveness(non_constructive)
        
        assert constructive_score > non_constructive_score