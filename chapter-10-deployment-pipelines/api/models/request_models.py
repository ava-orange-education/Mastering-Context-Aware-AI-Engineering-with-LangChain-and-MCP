"""
Pydantic models for API requests
"""

from pydantic import BaseModel, Field, validator
from typing import Optional, List
from enum import Enum


class AgentType(str, Enum):
    RETRIEVAL = "retrieval"
    ANALYSIS = "analysis"
    SYNTHESIS = "synthesis"
    MULTI_AGENT = "multi_agent"


class QueryRequest(BaseModel):
    """Request model for query endpoint"""
    
    query: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="User query to process"
    )
    
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of documents to retrieve"
    )
    
    agent_type: AgentType = Field(
        default=AgentType.MULTI_AGENT,
        description="Type of agent to use"
    )
    
    metadata: Optional[dict] = Field(
        default=None,
        description="Optional metadata for the query"
    )
    
    @validator('query')
    def validate_query(cls, v):
        if not v.strip():
            raise ValueError("Query cannot be empty or whitespace")
        return v.strip()


class DocumentUploadRequest(BaseModel):
    """Request model for document upload"""
    
    content: str = Field(
        ...,
        min_length=1,
        description="Document content"
    )
    
    metadata: Optional[dict] = Field(
        default=None,
        description="Document metadata"
    )
    
    chunk_size: int = Field(
        default=500,
        ge=100,
        le=2000,
        description="Size of text chunks"
    )
    
    chunk_overlap: int = Field(
        default=50,
        ge=0,
        le=500,
        description="Overlap between chunks"
    )


class FeedbackRequest(BaseModel):
    """Request model for user feedback"""
    
    query_id: str = Field(
        ...,
        description="ID of the query being rated"
    )
    
    rating: int = Field(
        ...,
        ge=1,
        le=5,
        description="Rating from 1 to 5"
    )
    
    comment: Optional[str] = Field(
        default=None,
        max_length=500,
        description="Optional feedback comment"
    )


class HealthCheckRequest(BaseModel):
    """Request model for detailed health check"""
    
    check_vector_store: bool = Field(
        default=True,
        description="Check vector store connectivity"
    )
    
    check_llm: bool = Field(
        default=True,
        description="Check LLM API connectivity"
    )
    
    check_agents: bool = Field(
        default=False,
        description="Check individual agents"
    )