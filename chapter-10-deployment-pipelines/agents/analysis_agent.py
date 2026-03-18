"""
Analysis agent for processing retrieved documents
"""

import anthropic
from typing import List, Dict, Any
import logging
import time

from vector_stores.base_store import SearchResult
from config.settings import settings

logger = logging.getLogger(__name__)


class AnalysisAgent:
    """Agent responsible for analyzing retrieved documents"""
    
    def __init__(self):
        self.client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        self.model = "claude-3-5-sonnet-20241022"
    
    async def analyze(
        self,
        query: str,
        documents: List[SearchResult]
    ) -> Dict[str, Any]:
        """Analyze retrieved documents in context of query"""
        start_time = time.time()
        
        try:
            # Prepare context from documents
            context = self._prepare_context(documents)
            
            # Create analysis prompt
            prompt = self._create_analysis_prompt(query, context)
            
            # Call Claude for analysis
            response = self.client.messages.create(
                model=self.model,
                max_tokens=2000,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            analysis = response.content[0].text
            
            analysis_time = time.time() - start_time
            
            logger.info(f"Analysis completed in {analysis_time:.3f}s")
            
            return {
                "analysis": analysis,
                "document_count": len(documents),
                "processing_time": analysis_time,
                "model": self.model
            }
            
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            raise
    
    def _prepare_context(self, documents: List[SearchResult]) -> str:
        """Prepare context string from documents"""
        context_parts = []
        
        for i, result in enumerate(documents, 1):
            doc = result.document
            context_parts.append(
                f"Document {i} (relevance: {result.score:.3f}):\n{doc.content}\n"
            )
        
        return "\n".join(context_parts)
    
    def _create_analysis_prompt(self, query: str, context: str) -> str:
        """Create prompt for analysis"""
        return f"""Analyze the following documents to answer the query.

Query: {query}

Retrieved Documents:
{context}

Provide a structured analysis that:
1. Identifies key information relevant to the query
2. Notes any contradictions or inconsistencies
3. Highlights important details
4. Assesses the overall relevance and quality of the documents

Analysis:"""
    
    async def health_check(self) -> bool:
        """Check if analysis agent is healthy"""
        try:
            # Test API connection with minimal request
            response = self.client.messages.create(
                model=self.model,
                max_tokens=10,
                messages=[{"role": "user", "content": "test"}]
            )
            return True
        except Exception as e:
            logger.error(f"Analysis agent health check failed: {e}")
            return False