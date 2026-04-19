"""
API Models for Education AI Tutor
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime


class StudentProfile(BaseModel):
    """Student profile model"""
    student_id: str
    grade_level: str
    learning_style: Optional[str] = "balanced"
    interests: List[str] = []
    mastery_levels: Dict[str, float] = {}
    learning_goals: List[str] = []


class ExplanationRequest(BaseModel):
    """Request for concept explanation"""
    student_id: str
    topic: str
    question: Optional[str] = None
    context: Optional[str] = None
    difficulty: Optional[str] = "medium"


class PracticeRequest(BaseModel):
    """Request for practice problems"""
    student_id: str
    topic: str
    difficulty: str = "medium"
    count: int = Field(default=5, ge=1, le=20)
    problem_types: Optional[List[str]] = None


class AssessmentRequest(BaseModel):
    """Request for assessment"""
    student_id: str
    topic: str
    questions: List[Dict[str, Any]]
    student_answers: List[Dict[str, Any]]


class FeedbackRequest(BaseModel):
    """Request for feedback on student work"""
    student_id: str
    student_work: Dict[str, Any]
    rubric: Optional[Dict[str, Any]] = None
    learning_goals: Optional[List[str]] = None


class LearningPathRequest(BaseModel):
    """Request for learning path generation"""
    student_id: str
    learning_goal: str
    target_mastery: float = 0.8
    time_budget: Optional[int] = None  # minutes
    deadline: Optional[datetime] = None


class ProgressRequest(BaseModel):
    """Request for progress report"""
    student_id: str
    days: int = 30
    include_details: bool = False


class ExplanationResponse(BaseModel):
    """Explanation response"""
    explanation: str
    topic: str
    difficulty: str
    learning_style: str
    teaching_elements: Dict[str, bool]
    timestamp: datetime


class PracticeResponse(BaseModel):
    """Practice problems response"""
    problems: List[Dict[str, Any]]
    topic: str
    difficulty: str
    total_count: int


class AssessmentResponse(BaseModel):
    """Assessment results response"""
    student_id: str
    topic: str
    overall_score: float
    mastery_level: float
    strengths: List[str]
    weaknesses: List[str]
    recommendations: List[str]
    detailed_results: List[Dict[str, Any]]


class FeedbackResponse(BaseModel):
    """Feedback response"""
    feedback: str
    feedback_components: Dict[str, bool]
    timestamp: datetime


class LearningPathResponse(BaseModel):
    """Learning path response"""
    learning_goal: str
    path: List[Dict[str, Any]]
    milestones: List[Dict[str, Any]]
    estimated_time_minutes: int
    estimated_days: int
    feasibility: Dict[str, Any]
    recommendations: List[str]


class ProgressResponse(BaseModel):
    """Progress report response"""
    student_id: str
    period_days: int
    total_sessions: int
    total_time_hours: float
    topics_practiced: int
    mastery_improvements: Dict[str, float]
    current_streak: int
    engagement_score: float
    recent_achievements: List[Dict[str, Any]]


class ErrorResponse(BaseModel):
    """Error response"""
    error_type: str
    error_message: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)