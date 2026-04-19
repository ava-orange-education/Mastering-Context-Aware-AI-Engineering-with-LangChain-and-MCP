"""
Document Search Agent

Searches across enterprise documents with permission awareness
"""

from typing import Dict, Any, List, Optional
import logging
from datetime import datetime
import sys
sys.path.append('../..')

from shared.base_agent import BaseAgent, AgentResponse
from rag.permission_aware_retriever import PermissionAwareRetriever
from access_control.permission_manager import PermissionManager

logger = logging.getLogger(__name__)


class DocumentSearchAgent(BaseAgent):
    """
    Agent for searching enterprise documents
    """
    
    def __init__(self):
        super().__init__(
            name="Document Search Agent",
            model="claude-sonnet-4-20250514",
            temperature=0.3
        )
        
        self.retriever = PermissionAwareRetriever()
        self.permission_manager = PermissionManager()
    
    def _get_system_prompt(self) -> str:
        """System prompt for document search"""
        return """You are an enterprise knowledge assistant helping employees find information.

Your role:
1. Search across company documents (SharePoint, Confluence, Slack, Drive)
2. Respect document permissions and access controls
3. Synthesize information from multiple sources
4. Provide accurate, cited responses
5. Identify relevant experts and stakeholders

Guidelines:
- Only return information the user has permission to access
- Cite specific documents and sources
- Note document recency and relevance
- Suggest related searches or documents
- Identify subject matter experts when relevant
- Maintain confidentiality of restricted documents

Output format:
- Direct answer to the query
- Source citations with document titles
- Related documents or searches
- Relevant experts (if applicable)
- Confidence level in the answer"""
    
    async def process(self, input_data: Dict[str, Any]) -> AgentResponse:
        """
        Process document search request
        
        Args:
            input_data: {
                "query": str,
                "user_id": str,
                "user_groups": List[str] (optional),
                "sources": List[str] (optional),
                "top_k": int (optional),
                "filters": Dict (optional)
            }
        
        Returns:
            AgentResponse with search results
        """
        
        query = input_data.get("query")
        user_id = input_data.get("user_id")
        user_groups = input_data.get("user_groups", [])
        sources = input_data.get("sources", ["all"])
        top_k = input_data.get("top_k", 10)
        filters = input_data.get("filters", {})
        
        logger.info(f"Searching documents for user {user_id}: {query}")
        
        # Initialize retriever
        await self.retriever.initialize()
        
        # Search with permissions
        search_results = await self.retriever.search(
            query=query,
            user_id=user_id,
            user_groups=user_groups,
            top_k=top_k,
            sources=sources,
            filters=filters
        )
        
        logger.info(f"Retrieved {len(search_results)} documents")
        
        # Check permissions for each document
        accessible_results = []
        for result in search_results:
            doc_id = result.document.id
            
            has_access = await self.permission_manager.check_access(
                user_id=user_id,
                document_id=doc_id,
                user_groups=user_groups
            )
            
            if has_access:
                accessible_results.append(result)
            else:
                logger.debug(f"User {user_id} denied access to {doc_id}")
        
        logger.info(f"User has access to {len(accessible_results)} documents")
        
        # Synthesize response
        response_text = await self._synthesize_response(
            query=query,
            search_results=accessible_results
        )
        
        # Extract metadata
        sources_cited = self._extract_sources(accessible_results)
        related_docs = self._identify_related_documents(accessible_results)
        
        return AgentResponse(
            content=response_text,
            agent_name=self.name,
            timestamp=datetime.utcnow(),
            metadata={
                "query": query,
                "user_id": user_id,
                "total_results": len(search_results),
                "accessible_results": len(accessible_results),
                "sources": sources_cited,
                "related_documents": related_docs
            },
            confidence=self._calculate_confidence(accessible_results),
            sources=[r.document.id for r in accessible_results]
        )
    
    async def _synthesize_response(
        self,
        query: str,
        search_results: List[Any]
    ) -> str:
        """Synthesize response from search results"""
        
        if not search_results:
            return "No accessible documents found matching your query. This may be because:\n1. No documents match your search\n2. Matching documents are restricted\n3. Try broader search terms or contact the document owner for access."
        
        # Format search results for LLM
        context = self._format_search_results(search_results)
        
        messages = [
            {
                "role": "user",
                "content": f"""Query: {query}

Retrieved Documents:
{context}

Synthesize a clear, accurate answer based on these documents. Include:
1. Direct answer to the query
2. Citations to specific documents
3. Any conflicting information noted
4. Related topics or documents worth exploring"""
            }
        ]
        
        response = await self._call_llm(messages)
        return response
    
    def _format_search_results(self, results: List[Any]) -> str:
        """Format search results for context"""
        
        formatted = []
        
        for i, result in enumerate(results[:5], 1):  # Top 5 results
            doc = result.document
            
            formatted.append(f"""
Document {i} (Relevance: {result.score:.2f}):
Title: {doc.metadata.get('title', 'Untitled')}
Source: {doc.metadata.get('source', 'Unknown')}
Modified: {doc.metadata.get('modified_date', 'Unknown')}
Author: {doc.metadata.get('author', 'Unknown')}

Content:
{doc.content[:500]}...
---
""")
        
        return "\n".join(formatted)
    
    def _extract_sources(self, results: List[Any]) -> List[Dict[str, str]]:
        """Extract source information"""
        
        sources = []
        
        for result in results[:5]:
            doc = result.document
            sources.append({
                "document_id": doc.id,
                "title": doc.metadata.get("title", "Untitled"),
                "source": doc.metadata.get("source", "Unknown"),
                "url": doc.metadata.get("url", ""),
                "relevance_score": round(result.score, 3)
            })
        
        return sources
    
    def _identify_related_documents(
        self,
        results: List[Any]
    ) -> List[str]:
        """Identify related documents"""
        
        # Documents with similar metadata or topics
        related = []
        
        if results:
            primary_doc = results[0].document
            primary_topics = primary_doc.metadata.get("topics", [])
            
            for result in results[1:6]:
                doc = result.document
                doc_topics = doc.metadata.get("topics", [])
                
                # Check topic overlap
                if any(topic in doc_topics for topic in primary_topics):
                    related.append(doc.id)
        
        return related
    
    def _calculate_confidence(self, results: List[Any]) -> float:
        """Calculate confidence in search results"""
        
        if not results:
            return 0.0
        
        # Confidence based on:
        # - Number of high-relevance results
        # - Average relevance score
        # - Document recency
        
        high_relevance = sum(1 for r in results if r.score > 0.8)
        avg_score = sum(r.score for r in results) / len(results)
        
        confidence = (
            (high_relevance / min(5, len(results))) * 0.5 +
            avg_score * 0.5
        )
        
        return round(confidence, 2)