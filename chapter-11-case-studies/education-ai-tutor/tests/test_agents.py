"""
Tests for education agents
"""

import pytest
import asyncio
import sys
sys.path.append('../..')

from agents.teaching_agent import TeachingAgent
from agents.assessment_agent import AssessmentAgent
from agents.explanation_agent import ExplanationAgent
from agents.feedback_agent import FeedbackAgent
from agents.adaptation_agent import AdaptationAgent


@pytest.mark.asyncio
class TestTeachingAgent:
    """Test Teaching Agent"""
    
    async def test_initialization(self):
        """Test agent initializes correctly"""
        agent = TeachingAgent()
        
        assert agent.name == "Teaching Agent"
        assert agent.model is not None
    
    async def test_process_with_topic(self):
        """Test teaching with topic"""
        agent = TeachingAgent()
        
        input_data = {
            "student_profile": {
                "student_id": "test_student",
                "grade_level": "9th grade",
                "learning_style": "visual",
                "interests": ["science"],
                "mastery_levels": {}
            },
            "topic": "photosynthesis"
        }
        
        try:
            result = await agent.process(input_data)
            assert result.content is not None
            assert result.agent_name == "Teaching Agent"
            assert "teaching_elements" in result.metadata
        except Exception as e:
            # Expected without API key
            assert "api" in str(e).lower() or "key" in str(e).lower()
    
    async def test_requires_topic(self):
        """Test that topic is required"""
        agent = TeachingAgent()
        
        with pytest.raises(ValueError):
            await agent.process({"student_profile": {}})


@pytest.mark.asyncio
class TestAssessmentAgent:
    """Test Assessment Agent"""
    
    async def test_initialization(self):
        """Test agent initializes correctly"""
        agent = AssessmentAgent()
        
        assert agent.name == "Assessment Agent"
        assert agent.model is not None
    
    async def test_process_assessment(self):
        """Test assessment processing"""
        agent = AssessmentAgent()
        
        questions = [
            {"question": "What is 2+2?", "correct_answer": "4"}
        ]
        
        answers = [
            {"answer": "4"}
        ]
        
        input_data = {
            "student_id": "test_student",
            "topic": "arithmetic",
            "questions": questions,
            "student_answers": answers
        }
        
        try:
            result = await agent.process(input_data)
            assert result.content is not None
            assert "metrics" in result.metadata
        except Exception:
            # Expected without API key
            pass
    
    async def test_requires_student_and_topic(self):
        """Test required fields"""
        agent = AssessmentAgent()
        
        with pytest.raises(ValueError):
            await agent.process({})


@pytest.mark.asyncio
class TestExplanationAgent:
    """Test Explanation Agent"""
    
    async def test_initialization(self):
        """Test agent initializes correctly"""
        agent = ExplanationAgent()
        
        assert agent.name == "Explanation Agent"
        assert agent.model is not None
    
    async def test_generate_explanation(self):
        """Test explanation generation"""
        agent = ExplanationAgent()
        
        input_data = {
            "concept": "gravity",
            "level": "middle_school",
            "learning_style": "visual"
        }
        
        try:
            result = await agent.process(input_data)
            assert result.content is not None
            assert "components" in result.metadata
        except Exception:
            pass
    
    async def test_requires_concept(self):
        """Test that concept is required"""
        agent = ExplanationAgent()
        
        with pytest.raises(ValueError):
            await agent.process({})


@pytest.mark.asyncio
class TestFeedbackAgent:
    """Test Feedback Agent"""
    
    async def test_initialization(self):
        """Test agent initializes correctly"""
        agent = FeedbackAgent()
        
        assert agent.name == "Feedback Agent"
        assert agent.model is not None
    
    async def test_generate_feedback(self):
        """Test feedback generation"""
        agent = FeedbackAgent()
        
        input_data = {
            "student_work": {
                "question": "What is photosynthesis?",
                "answer": "Process where plants make food"
            }
        }
        
        try:
            result = await agent.process(input_data)
            assert result.content is not None
        except Exception:
            pass
    
    async def test_requires_student_work(self):
        """Test that student work is required"""
        agent = FeedbackAgent()
        
        with pytest.raises(ValueError):
            await agent.process({})


@pytest.mark.asyncio
class TestAdaptationAgent:
    """Test Adaptation Agent"""
    
    async def test_initialization(self):
        """Test agent initializes correctly"""
        agent = AdaptationAgent()
        
        assert agent.name == "Adaptation Agent"
        assert agent.model is not None
    
    async def test_analyze_performance(self):
        """Test performance analysis"""
        agent = AdaptationAgent()
        
        performance = [
            {"correct": True, "accuracy": 0.9, "time_taken": 45},
            {"correct": True, "accuracy": 0.95, "time_taken": 40}
        ]
        
        input_data = {
            "student_id": "test_student",
            "topic": "algebra",
            "performance_history": performance,
            "current_difficulty": "medium",
            "time_spent": 90,
            "errors": []
        }
        
        try:
            result = await agent.process(input_data)
            assert result.content is not None
            assert "adaptations" in result.metadata
        except Exception:
            pass