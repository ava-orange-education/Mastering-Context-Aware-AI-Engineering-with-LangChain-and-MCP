"""
Pydantic models for API responses
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


class ResponseStatus(str, Enum):
    SUCCESS = "success"
    ERROR = "error"
    PARTIAL = "partial"


class DocumentMetadata(BaseModel):
    """Metadata for retrieved documents"""
    
    id: str
    score: float
    source: Optional[str] = None
    chunk_index: Optional[int] = None


class RetrievedDocument(BaseModel):
    """Retrieved document with content and metadata"""
    
    content: str
    metadata: DocumentMetadata


class QueryResponse(BaseModel):
    """Response model for query endpoint"""
    
    status: ResponseStatus
    query: str
    response: str
    
    retrieved_documents: Optional[List[RetrievedDocument]] = None
    
    agent_type: str
    processing_time_ms: float
    
    evaluation_scores: Optional[Dict[str, float]] = None
    
    query_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ErrorResponse(BaseModel):
    """Error response model"""
    
    status: ResponseStatus = ResponseStatus.ERROR
    error_type: str
    error_message: str
    detail: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class HealthStatus(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


class ComponentHealth(BaseModel):
    """Health status of individual component"""
    
    status: HealthStatus
    message: Optional[str] = None
    latency_ms: Optional[float] = None


class HealthCheckResponse(BaseModel):
    """Response model for health check"""
    
    status: HealthStatus
    version: str
    uptime_seconds: float
    
    components: Dict[str, ComponentHealth]
    
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class DocumentUploadResponse(BaseModel):
    """Response model for document upload"""
    
    status: ResponseStatus
    document_id: str
    chunks_created: int
    processing_time_ms: float
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class FeedbackResponse(BaseModel):
    """Response model for feedback submission"""
    
    status: ResponseStatus
    feedback_id: str
    message: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class MetricsResponse(BaseModel):
    """Response model for metrics endpoint"""
    
    total_queries: int
    average_response_time_ms: float
    success_rate: float
    
    queries_last_hour: int
    queries_last_24h: int
    
    agent_usage: Dict[str, int]
    
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }