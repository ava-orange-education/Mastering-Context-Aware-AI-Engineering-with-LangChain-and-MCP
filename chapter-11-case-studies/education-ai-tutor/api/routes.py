"""
API Routes for Education AI Tutor
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any
import logging
import sys
sys.path.append('../..')

from .models import (
    ExplanationRequest,
    ExplanationResponse,
    PracticeRequest,
    PracticeResponse,
    AssessmentRequest,
    AssessmentResponse,
    FeedbackRequest,
    FeedbackResponse,
    LearningPathRequest,
    LearningPathResponse,
    ProgressRequest,
    ProgressResponse
)
from agents.teaching_agent import TeachingAgent
from agents.assessment_agent import AssessmentAgent
from agents.feedback_agent import FeedbackAgent
from student_model.knowledge_state import KnowledgeState
from student_model.progress_tracker import ProgressTracker
from personalization.learning_path_generator import LearningPathGenerator

logger = logging.getLogger(__name__)
router = APIRouter()

# Initialize components
teaching_agent = TeachingAgent()
assessment_agent = AssessmentAgent()
feedback_agent = FeedbackAgent()

# In-memory storage (use database in production)
student_knowledge_states: Dict[str, KnowledgeState] = {}
student_progress_trackers: Dict[str, ProgressTracker] = {}
path_generator = LearningPathGenerator()


def get_or_create_knowledge_state(student_id: str) -> KnowledgeState:
    """Get or create knowledge state for student"""
    if student_id not in student_knowledge_states:
        student_knowledge_states[student_id] = KnowledgeState(student_id)
    return student_knowledge_states[student_id]


def get_or_create_progress_tracker(student_id: str) -> ProgressTracker:
    """Get or create progress tracker for student"""
    if student_id not in student_progress_trackers:
        student_progress_trackers[student_id] = ProgressTracker(student_id)
    return student_progress_trackers[student_id]


@router.post("/explain", response_model=ExplanationResponse)
async def get_explanation(request: ExplanationRequest):
    """
    Get personalized explanation for a concept
    """
    
    try:
        logger.info(f"Explanation request for {request.topic} by {request.student_id}")
        
        # Get student knowledge state
        knowledge_state = get_or_create_knowledge_state(request.student_id)
        
        # Prepare student profile
        student_profile = {
            "student_id": request.student_id,
            "grade_level": "high_school",  # Should come from student profile
            "learning_style": "visual",
            "mastery_levels": knowledge_state.mastery_levels
        }
        
        # Get explanation
        result = await teaching_agent.process({
            "student_profile": student_profile,
            "topic": request.topic,
            "question": request.question,
            "context": request.context
        })
        
        return ExplanationResponse(
            explanation=result.content,
            topic=request.topic,
            difficulty=request.difficulty,
            learning_style=student_profile["learning_style"],
            teaching_elements=result.metadata.get("teaching_elements", {}),
            timestamp=result.timestamp
        )
        
    except Exception as e:
        logger.error(f"Error generating explanation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/practice", response_model=PracticeResponse)
async def generate_practice(request: PracticeRequest):
    """
    Generate practice problems
    """
    
    try:
        logger.info(f"Practice request for {request.topic} by {request.student_id}")
        
        # Generate practice problems
        # In production, use PracticeGeneratorAgent
        problems = []
        
        for i in range(request.count):
            problems.append({
                "id": f"prob_{i+1}",
                "question": f"Practice problem {i+1} on {request.topic}",
                "difficulty": request.difficulty,
                "topic": request.topic,
                "hints": ["Hint 1", "Hint 2"],
                "solution": f"Solution for problem {i+1}"
            })
        
        return PracticeResponse(
            problems=problems,
            topic=request.topic,
            difficulty=request.difficulty,
            total_count=len(problems)
        )
        
    except Exception as e:
        logger.error(f"Error generating practice: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/assess", response_model=AssessmentResponse)
async def assess_student(request: AssessmentRequest):
    """
    Assess student understanding
    """
    
    try:
        logger.info(f"Assessment for {request.topic} by {request.student_id}")
        
        # Perform assessment
        result = await assessment_agent.process({
            "student_id": request.student_id,
            "topic": request.topic,
            "questions": request.questions,
            "student_answers": request.student_answers
        })
        
        # Update knowledge state
        knowledge_state = get_or_create_knowledge_state(request.student_id)
        
        metrics = result.metadata.get("metrics", {})
        mastery_level = metrics.get("mastery_level", 0.5)
        
        knowledge_state.update_mastery(
            concept=request.topic,
            performance={
                "correct": metrics.get("accuracy_rate", 0) > 0.7,
                "difficulty": "medium",
                "time_taken": 60,
                "attempts": 1
            }
        )
        
        # Extract recommendations
        evaluations = result.metadata.get("evaluations", [])
        all_strengths = []
        all_weaknesses = []
        
        for eval in evaluations:
            all_strengths.extend(eval.get("strengths", []))
            all_weaknesses.extend(eval.get("weaknesses", []))
        
        return AssessmentResponse(
            student_id=request.student_id,
            topic=request.topic,
            overall_score=metrics.get("average_score", 0),
            mastery_level=mastery_level,
            strengths=list(set(all_strengths))[:5],
            weaknesses=list(set(all_weaknesses))[:5],
            recommendations=["Practice more problems", "Review weak areas"],
            detailed_results=evaluations
        )
        
    except Exception as e:
        logger.error(f"Error in assessment: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/feedback", response_model=FeedbackResponse)
async def get_feedback(request: FeedbackRequest):
    """
    Get feedback on student work
    """
    
    try:
        logger.info(f"Feedback request for {request.student_id}")
        
        result = await feedback_agent.process({
            "student_work": request.student_work,
            "rubric": request.rubric,
            "learning_goals": request.learning_goals or []
        })
        
        return FeedbackResponse(
            feedback=result.content,
            feedback_components=result.metadata.get("components", {}),
            timestamp=result.timestamp
        )
        
    except Exception as e:
        logger.error(f"Error generating feedback: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/learning-path", response_model=LearningPathResponse)
async def create_learning_path(request: LearningPathRequest):
    """
    Generate personalized learning path
    """
    
    try:
        logger.info(f"Learning path request for {request.student_id}")
        
        # Get current knowledge
        knowledge_state = get_or_create_knowledge_state(request.student_id)
        
        # Generate path
        path_result = path_generator.generate_learning_path(
            student_id=request.student_id,
            learning_goal=request.learning_goal,
            current_knowledge=knowledge_state.mastery_levels,
            target_mastery=request.target_mastery,
            time_budget=request.time_budget,
            deadline=request.deadline
        )
        
        return LearningPathResponse(**path_result)
        
    except Exception as e:
        logger.error(f"Error generating learning path: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/progress/{student_id}", response_model=ProgressResponse)
async def get_progress(student_id: str, days: int = 30):
    """
    Get student progress report
    """
    
    try:
        logger.info(f"Progress report for {student_id}")
        
        # Get progress tracker
        tracker = get_or_create_progress_tracker(student_id)
        
        # Get progress summary
        summary = tracker.get_progress_summary(days=days)
        
        # Get knowledge state for mastery improvements
        knowledge_state = get_or_create_knowledge_state(student_id)
        
        return ProgressResponse(
            student_id=student_id,
            period_days=days,
            total_sessions=summary.get("total_sessions", 0),
            total_time_hours=summary.get("total_time_hours", 0),
            topics_practiced=summary.get("subjects_practiced", 0),
            mastery_improvements=knowledge_state.mastery_levels,
            current_streak=summary.get("current_streak", 0),
            engagement_score=0.75,  # From engagement metrics
            recent_achievements=summary.get("recent_milestones", [])
        )
        
    except Exception as e:
        logger.error(f"Error getting progress: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/session")
async def record_session(
    student_id: str,
    topic: str,
    duration: float,
    problems_attempted: int,
    problems_correct: int
):
    """
    Record a learning session
    """
    
    try:
        tracker = get_or_create_progress_tracker(student_id)
        
        tracker.record_session(
            subject="general",
            topic=topic,
            duration=duration,
            problems_attempted=problems_attempted,
            problems_correct=problems_correct,
            concepts_covered=[topic],
            difficulty_level="medium"
        )
        
        return {"status": "success", "message": "Session recorded"}
        
    except Exception as e:
        logger.error(f"Error recording session: {e}")
        raise HTTPException(status_code=500, detail=str(e))