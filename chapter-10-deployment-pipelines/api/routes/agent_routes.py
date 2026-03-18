"""
Agent interaction routes
"""

from fastapi import APIRouter, Depends, HTTPException, status
from typing import Dict, Optional
import logging
import time
import uuid

from api.models.request_models import QueryRequest, AgentType
from api.models.response_models import (
    QueryResponse,
    ResponseStatus,
    RetrievedDocument,
    DocumentMetadata
)
from api.dependencies import (
    get_retrieval_agent,
    get_analysis_agent,
    get_synthesis_agent,
    get_current_user_optional
)
from api.middleware.rate_limit_middleware import limiter, RateLimitConfig
from agents import RetrievalAgent, AnalysisAgent, SynthesisAgent

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/query", response_model=QueryResponse)
@limiter.limit(RateLimitConfig.QUERY)
async def process_query(
    request: QueryRequest,
    retrieval_agent: RetrievalAgent = Depends(get_retrieval_agent),
    analysis_agent: AnalysisAgent = Depends(get_analysis_agent),
    synthesis_agent: SynthesisAgent = Depends(get_synthesis_agent),
    user: Optional[Dict] = Depends(get_current_user_optional)
):
    """Process a query through the multi-agent pipeline"""
    query_id = str(uuid.uuid4())
    start_time = time.time()
    
    try:
        logger.info(
            f"Processing query {query_id}: '{request.query}' "
            f"(agent_type: {request.agent_type})"
        )
        
        if request.agent_type == AgentType.MULTI_AGENT:
            # Full multi-agent pipeline
            
            # Step 1: Retrieve documents
            search_results = await retrieval_agent.retrieve(
                query=request.query,
                top_k=request.top_k
            )
            
            # Convert to response format
            retrieved_docs = [
                RetrievedDocument(
                    content=result.document.content,
                    metadata=DocumentMetadata(
                        id=result.document.id,
                        score=result.score,
                        source=result.document.metadata.get("source") 
                            if result.document.metadata else None
                    )
                )
                for result in search_results
            ]
            
            # Step 2: Analyze documents
            analysis_result = await analysis_agent.analyze(
                query=request.query,
                documents=search_results
            )
            
            # Step 3: Synthesize response
            context = "\n\n".join([doc.content for doc in retrieved_docs])
            synthesis_result = await synthesis_agent.synthesize(
                query=request.query,
                analysis=analysis_result,
                context=context
            )
            
            response_text = synthesis_result["response"]
        
        elif request.agent_type == AgentType.RETRIEVAL:
            # Retrieval only
            search_results = await retrieval_agent.retrieve(
                query=request.query,
                top_k=request.top_k
            )
            
            retrieved_docs = [
                RetrievedDocument(
                    content=result.document.content,
                    metadata=DocumentMetadata(
                        id=result.document.id,
                        score=result.score
                    )
                )
                for result in search_results
            ]
            
            response_text = f"Retrieved {len(retrieved_docs)} documents"
        
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported agent type: {request.agent_type}"
            )
        
        processing_time = (time.time() - start_time) * 1000
        
        logger.info(
            f"Query {query_id} completed in {processing_time:.2f}ms"
        )
        
        return QueryResponse(
            status=ResponseStatus.SUCCESS,
            query=request.query,
            response=response_text,
            retrieved_documents=retrieved_docs if request.agent_type != AgentType.SYNTHESIS else None,
            agent_type=request.agent_type.value,
            processing_time_ms=processing_time,
            query_id=query_id
        )
    
    except Exception as e:
        logger.error(f"Query {query_id} failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query processing failed: {str(e)}"
        )


@router.get("/query/{query_id}")
async def get_query_status(
    query_id: str,
    user: Optional[Dict] = Depends(get_current_user_optional)
):
    """Get status of a query (placeholder for async queries)"""
    # In a real implementation, this would check a database/cache
    # for the query status
    return {
        "query_id": query_id,
        "status": "completed",
        "message": "Query status endpoint (not yet implemented)"
    }


@router.post("/query/batch")
@limiter.limit("10/minute")
async def process_batch_queries(
    queries: list[QueryRequest],
    retrieval_agent: RetrievalAgent = Depends(get_retrieval_agent),
    analysis_agent: AnalysisAgent = Depends(get_analysis_agent),
    synthesis_agent: SynthesisAgent = Depends(get_synthesis_agent),
    user: Optional[Dict] = Depends(get_current_user_optional)
):
    """Process multiple queries in batch"""
    
    if len(queries) > 10:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Maximum 10 queries per batch"
        )
    
    results = []
    
    for query_request in queries:
        try:
            # Process each query (reusing the single query logic)
            result = await process_query(
                request=query_request,
                retrieval_agent=retrieval_agent,
                analysis_agent=analysis_agent,
                synthesis_agent=synthesis_agent,
                user=user
            )
            results.append(result)
        
        except Exception as e:
            logger.error(f"Batch query failed: {e}")
            results.append({
                "status": "error",
                "query": query_request.query,
                "error": str(e)
            })
    
    return {
        "status": "completed",
        "total_queries": len(queries),
        "results": results
    }