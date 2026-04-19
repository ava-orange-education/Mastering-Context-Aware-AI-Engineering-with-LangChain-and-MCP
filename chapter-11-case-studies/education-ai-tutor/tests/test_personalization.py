"""
Tests for personalization components
"""

import pytest
import sys
sys.path.append('../..')

from personalization.content_recommender import ContentRecommender
from personalization.difficulty_scaler import DifficultyScaler
from personalization.learning_path_generator import LearningPathGenerator


class TestContentRecommender:
    """Test Content Recommender"""
    
    def test_initialization(self):
        """Test recommender initializes"""
        recommender = ContentRecommender()
        
        assert recommender.content_effectiveness is not None
        assert recommender.student_preferences is not None
    
    def test_recommend_content(self):
        """Test content recommendation"""
        recommender = ContentRecommender()
        
        recommendations = recommender.recommend_content(
            student_id="test_student",
            topic="algebra",
            learning_style="visual",
            mastery_level=0.6,
            interests=["science"],
            previous_content=[],
            max_recommendations=5
        )
        
        assert len(recommendations) > 0
        assert len(recommendations) <= 5
        assert all("score" in rec for rec in recommendations)
    
    def test_record_effectiveness(self):
        """Test recording content effectiveness"""
        recommender = ContentRecommender()
        
        recommender.record_content_effectiveness(
            student_id="test_student",
            content_type="video",
            effectiveness=0.85
        )
        
        key = "test_student_video"
        assert key in recommender.content_effectiveness
        assert 0.85 in recommender.content_effectiveness[key]


class TestDifficultyScaler:
    """Test Difficulty Scaler"""
    
    def test_initialization(self):
        """Test scaler initializes"""
        scaler = DifficultyScaler()
        
        assert scaler.target_accuracy == 0.75
        assert scaler.performance_history is not None
    
    def test_calculate_next_difficulty(self):
        """Test difficulty calculation"""
        scaler = DifficultyScaler()
        
        # High performance - should increase difficulty
        high_performance = [
            {"accuracy": 0.95, "time_taken": 40, "attempts": 1, "certainty": 0.9}
            for _ in range(5)
        ]
        
        result = scaler.calculate_next_difficulty(
            student_id="test_student",
            current_difficulty="medium",
            recent_performance=high_performance,
            topic="algebra"
        )
        
        assert result["adjustment"] in ["increase", "maintain", "decrease"]
        assert "recommended_difficulty" in result
        assert "confidence" in result
    
    def test_low_performance_decreases_difficulty(self):
        """Test that low performance decreases difficulty"""
        scaler = DifficultyScaler()
        
        low_performance = [
            {"accuracy": 0.3, "time_taken": 120, "attempts": 3, "certainty": 0.2}
            for _ in range(5)
        ]
        
        result = scaler.calculate_next_difficulty(
            student_id="test_student",
            current_difficulty="hard",
            recent_performance=low_performance,
            topic="algebra"
        )
        
        assert result["adjustment"] == "decrease"
    
    def test_optimal_difficulty_distribution(self):
        """Test optimal difficulty distribution"""
        scaler = DifficultyScaler()
        
        distribution = scaler.get_optimal_difficulty_distribution(mastery_level=0.6)
        
        assert isinstance(distribution, dict)
        assert abs(sum(distribution.values()) - 1.0) < 0.01  # Should sum to 1.0


class TestLearningPathGenerator:
    """Test Learning Path Generator"""
    
    def test_initialization(self):
        """Test generator initializes"""
        generator = LearningPathGenerator()
        
        assert generator.concept_graph is not None
        assert generator.concept_times is not None
    
    def test_add_concept_dependency(self):
        """Test adding concept dependency"""
        generator = LearningPathGenerator()
        
        generator.add_concept_dependency(
            concept="algebra",
            prerequisites=["arithmetic"],
            estimated_time=60,
            difficulty="medium"
        )
        
        assert "algebra" in generator.concept_graph
        assert generator.concept_graph["algebra"] == ["arithmetic"]
        assert generator.concept_times["algebra"] == 60
    
    def test_generate_learning_path(self):
        """Test learning path generation"""
        generator = LearningPathGenerator()
        
        # Set up dependencies
        generator.add_concept_dependency("addition", [], 30, "beginner")
        generator.add_concept_dependency("multiplication", ["addition"], 45, "easy")
        generator.add_concept_dependency("division", ["multiplication"], 50, "medium")
        
        # Current knowledge
        current_knowledge = {
            "addition": 0.9,
            "multiplication": 0.5
        }
        
        # Generate path
        path = generator.generate_learning_path(
            student_id="test_student",
            learning_goal="division",
            current_knowledge=current_knowledge,
            target_mastery=0.8
        )
        
        assert "learning_goal" in path
        assert path["learning_goal"] == "division"
        assert "path" in path
        assert "estimated_time_minutes" in path
        assert len(path["path"]) > 0
    
    def test_feasibility_assessment(self):
        """Test feasibility assessment"""
        generator = LearningPathGenerator()
        
        feasibility = generator._assess_feasibility(
            total_time=120,
            time_budget=100,
            deadline=None
        )
        
        assert "feasible" in feasibility
        assert feasibility["feasible"] is False  # Needs more time than available
        assert len(feasibility["issues"]) > 0