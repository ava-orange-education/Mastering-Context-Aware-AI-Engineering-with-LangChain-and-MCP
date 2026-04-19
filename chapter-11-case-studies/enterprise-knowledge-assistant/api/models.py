"""
API models for Enterprise Knowledge Assistant
"""

from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum


class SearchRequest(BaseModel):
    """Request for document search"""
    query: str = Field(..., min_length=1, description="Search query")
    user_id: str = Field(..., description="User performing search")
    user_groups: Optional[List[str]] = Field(default=None, description="User's group memberships")
    sources: Optional[List[str]] = Field(default=["all"], description="Sources to search")
    top_k: int = Field(10, ge=1, le=50, description="Number of results")
    filters: Optional[Dict[str, Any]] = Field(default=None, description="Additional filters")


class DocumentResult(BaseModel):
    """Search result document"""
    document_id: str
    title: str
    content: str
    source: str
    url: Optional[str] = None
    modified_date: Optional[str] = None
    author: Optional[str] = None
    relevance_score: float


class SearchResponse(BaseModel):
    """Response from document search"""
    status: str = "success"
    query: str
    results: List[DocumentResult]
    total_results: int
    accessible_results: int
    response_time: float
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class SummarizeRequest(BaseModel):
    """Request for document summarization"""
    document_id: Optional[str] = None
    content: Optional[str] = None
    document_type: str = "document"
    max_length: int = Field(500, ge=100, le=2000)
    focus_areas: Optional[List[str]] = None
    
    @validator('content', 'document_id')
    def check_content_or_id(cls, v, values):
        if 'document_id' not in values and 'content' not in values:
            if not v:
                raise ValueError("Either document_id or content must be provided")
        return v


class SummarizeResponse(BaseModel):
    """Response from summarization"""
    status: str = "success"
    summary: str
    document_type: str
    original_length: int
    summary_length: int
    compression_ratio: float
    structured_info: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class RelatedDocumentsRequest(BaseModel):
    """Request for related documents"""
    document_id: Optional[str] = None
    query: Optional[str] = None
    user_id: str
    relationship_types: List[str] = Field(default=["all"])
    max_results: int = Field(10, ge=1, le=50)
    
    @validator('query', 'document_id')
    def check_query_or_id(cls, v, values):
        if 'document_id' not in values and 'query' not in values:
            if not v:
                raise ValueError("Either document_id or query must be provided")
        return v


class RelationshipInfo(BaseModel):
    """Document relationship information"""
    document_id: str
    title: str
    source: str
    relationship_type: str
    relationship_strength: str
    relevance_score: float
    explanation: str


class RelatedDocumentsResponse(BaseModel):
    """Response with related documents"""
    status: str = "success"
    primary_document: Optional[str] = None
    query: Optional[str] = None
    relationships: List[RelationshipInfo]
    total_related: int
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ExpertFinderRequest(BaseModel):
    """Request to find experts"""
    topic: str = Field(..., min_length=1)
    department: Optional[str] = None
    min_contributions: int = Field(3, ge=1)
    recency_days: int = Field(365, ge=30, le=1095)
    max_experts: int = Field(5, ge=1, le=20)


class ExpertInfo(BaseModel):
    """Expert information"""
    author: str
    email: Optional[str] = None
    department: Optional[str] = None
    expertise_level: str
    contributions: int
    recent_contributions: int
    expertise_score: float
    key_contributions: List[str]


class ExpertFinderResponse(BaseModel):
    """Response with expert recommendations"""
    status: str = "success"
    topic: str
    experts: List[ExpertInfo]
    documents_analyzed: int
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class FeedbackRequest(BaseModel):
    """User feedback on search results"""
    query_id: str
    user_id: str
    rating: int = Field(..., ge=0, le=5)
    feedback_text: Optional[str] = None
    feedback_type: str = Field("rating", pattern="^(rating|thumbs|comment)$")


class FeedbackResponse(BaseModel):
    """Feedback submission response"""
    status: str = "success"
    message: str = "Feedback recorded"
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class MetricsResponse(BaseModel):
    """System metrics response"""
    total_queries: int
    avg_response_time: float
    avg_results_count: float
    satisfaction_score: float
    click_through_rate: float
    period_days: int
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ErrorResponse(BaseModel):
    """Error response"""
    status: str = "error"
    error_type: str
    error_message: str
    detail: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)