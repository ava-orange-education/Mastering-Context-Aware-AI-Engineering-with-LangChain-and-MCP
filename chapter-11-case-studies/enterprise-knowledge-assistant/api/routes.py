"""
API routes for Enterprise Knowledge Assistant
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any
import logging
from datetime import datetime
import time
import uuid
import sys
sys.path.append('../..')

from .models import (
    SearchRequest, SearchResponse, DocumentResult,
    SummarizeRequest, SummarizeResponse,
    RelatedDocumentsRequest, RelatedDocumentsResponse, RelationshipInfo,
    ExpertFinderRequest, ExpertFinderResponse, ExpertInfo,
    FeedbackRequest, FeedbackResponse,
    MetricsResponse, ErrorResponse
)
from agents.document_search_agent import DocumentSearchAgent
from agents.summarization_agent import SummarizationAgent
from agents.cross_reference_agent import CrossReferenceAgent
from agents.expert_finder_agent import ExpertFinderAgent
from evaluation.user_satisfaction import UserSatisfactionMetrics

logger = logging.getLogger(__name__)
router = APIRouter()

# Initialize agents
search_agent = DocumentSearchAgent()
summarization_agent = SummarizationAgent()
cross_ref_agent = CrossReferenceAgent()
expert_agent = ExpertFinderAgent()
satisfaction_metrics = UserSatisfactionMetrics()


@router.post("/search", response_model=SearchResponse)
async def search_documents(request: SearchRequest):
    """
    Search across enterprise documents
    
    Searches multiple sources with permission awareness and
    returns relevant documents.
    """
    
    start_time = time.time()
    query_id = str(uuid.uuid4())
    
    try:
        # Process search
        result = await search_agent.process({
            "query": request.query,
            "user_id": request.user_id,
            "user_groups": request.user_groups,
            "sources": request.sources,
            "top_k": request.top_k,
            "filters": request.filters
        })
        
        # Convert to response format
        document_results = [
            DocumentResult(
                document_id=source,
                title=result.metadata.get("sources", [{}])[i].get("title", "Untitled"),
                content=result.content[:500],  # Preview
                source=result.metadata.get("sources", [{}])[i].get("source", "Unknown"),
                url=result.metadata.get("sources", [{}])[i].get("url"),
                relevance_score=result.metadata.get("sources", [{}])[i].get("relevance_score", 0.0)
            )
            for i, source in enumerate(result.sources[:request.top_k])
        ]
        
        response_time = time.time() - start_time
        
        # Log query
        satisfaction_metrics.log_query(
            query_id=query_id,
            user_id=request.user_id,
            query=request.query,
            results_count=len(document_results),
            response_time=response_time
        )
        
        return SearchResponse(
            query=request.query,
            results=document_results,
            total_results=result.metadata.get("total_results", 0),
            accessible_results=result.metadata.get("accessible_results", 0),
            response_time=round(response_time, 3)
        )
    
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/summarize", response_model=SummarizeResponse)
async def summarize_document(request: SummarizeRequest):
    """
    Summarize a document or content
    
    Creates concise summaries of documents, meetings, or conversations.
    """
    
    try:
        # Get content
        if request.content:
            content = request.content
        elif request.document_id:
            # In production, fetch document content
            content = "Sample document content"
        else:
            raise HTTPException(status_code=400, detail="Either content or document_id required")
        
        # Summarize
        result = await summarization_agent.process({
            "content": content,
            "document_type": request.document_type,
            "max_length": request.max_length,
            "focus_areas": request.focus_areas
        })
        
        return SummarizeResponse(
            summary=result.content,
            document_type=request.document_type,
            original_length=result.metadata.get("original_length", 0),
            summary_length=result.metadata.get("summary_length", 0),
            compression_ratio=result.metadata.get("compression_ratio", 0.0),
            structured_info=result.metadata.get("structured_info", {})
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Summarization failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/related", response_model=RelatedDocumentsResponse)
async def find_related_documents(request: RelatedDocumentsRequest):
    """
    Find related documents
    
    Discovers documents related to a given document or query
    through various relationship types.
    """
    
    try:
        result = await cross_ref_agent.process({
            "document_id": request.document_id,
            "query": request.query,
            "user_id": request.user_id,
            "relationship_types": request.relationship_types,
            "max_results": request.max_results
        })
        
        # Convert to response format
        relationships = [
            RelationshipInfo(
                document_id=rel["document_id"],
                title=rel["title"],
                source=rel["source"],
                relationship_type=rel["relationship_type"],
                relationship_strength=rel["relationship_strength"],
                relevance_score=rel["relevance_score"],
                explanation=rel["explanation"]
            )
            for rel in result.metadata.get("relationships", [])
        ]
        
        return RelatedDocumentsResponse(
            primary_document=request.document_id,
            query=request.query,
            relationships=relationships,
            total_related=len(relationships)
        )
    
    except Exception as e:
        logger.error(f"Related documents search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/experts", response_model=ExpertFinderResponse)
async def find_experts(request: ExpertFinderRequest):
    """
    Find subject matter experts
    
    Identifies experts based on document authorship and contributions.
    """
    
    try:
        result = await expert_agent.process({
            "topic": request.topic,
            "department": request.department,
            "min_contributions": request.min_contributions,
            "recency_days": request.recency_days,
            "max_experts": request.max_experts
        })
        
        # Convert to response format
        experts = [
            ExpertInfo(
                author=exp["author"],
                email=exp.get("email"),
                department=exp.get("department"),
                expertise_level=exp["expertise_level"],
                contributions=len(exp["contributions"]),
                recent_contributions=exp["recent_contributions"],
                expertise_score=exp["expertise_score"],
                key_contributions=[c["title"] for c in exp["contributions"][:3]]
            )
            for exp in result.metadata.get("experts", [])
        ]
        
        return ExpertFinderResponse(
            topic=request.topic,
            experts=experts,
            documents_analyzed=result.metadata.get("documents_analyzed", 0)
        )
    
    except Exception as e:
        logger.error(f"Expert finding failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(request: FeedbackRequest):
    """
    Submit user feedback
    
    Records user satisfaction with search results.
    """
    
    try:
        satisfaction_metrics.log_feedback(
            query_id=request.query_id,
            user_id=request.user_id,
            rating=request.rating,
            feedback_text=request.feedback_text,
            feedback_type=request.feedback_type
        )
        
        return FeedbackResponse(
            message=f"Feedback recorded for query {request.query_id}"
        )
    
    except Exception as e:
        logger.error(f"Feedback submission failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics", response_model=MetricsResponse)
async def get_metrics(days: int = 7):
    """Get system metrics"""
    
    try:
        metrics = satisfaction_metrics.calculate_metrics(days=days)
        
        return MetricsResponse(
            total_queries=metrics["total_queries"],
            avg_response_time=metrics["avg_response_time"],
            avg_results_count=metrics["avg_results_count"],
            satisfaction_score=metrics["satisfaction_score"],
            click_through_rate=metrics["click_through_rate"],
            period_days=days
        )
    
    except Exception as e:
        logger.error(f"Failed to get metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))