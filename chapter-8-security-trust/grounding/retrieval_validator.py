"""
Validate retrieval quality and relevance.
"""

from typing import Dict, Any, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RetrievalValidator:
    """Validate quality of retrieval results"""
    
    def __init__(self, llm_client):
        """
        Initialize retrieval validator
        
        Args:
            llm_client: LLM client for validation
        """
        self.llm = llm_client
    
    def validate_relevance(self, query: str, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Validate that retrieved documents are relevant to query
        
        Args:
            query: Search query
            documents: Retrieved documents
            
        Returns:
            Relevance validation result
        """
        if not documents:
            return {
                'is_relevant': False,
                'relevance_score': 0.0,
                'reason': 'No documents retrieved'
            }
        
        # Simple keyword overlap check
        query_words = set(query.lower().split())
        
        relevance_scores = []
        for doc in documents:
            content = doc.get('content', '')
            doc_words = set(content.lower().split())
            
            # Calculate overlap
            overlap = len(query_words & doc_words)
            score = overlap / max(len(query_words), 1)
            relevance_scores.append(score)
        
        avg_relevance = sum(relevance_scores) / len(relevance_scores)
        
        return {
            'is_relevant': avg_relevance >= 0.3,  # 30% overlap threshold
            'relevance_score': avg_relevance,
            'documents_checked': len(documents),
            'individual_scores': relevance_scores
        }
    
    def validate_coverage(self, query: str, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Check if documents adequately cover the query
        
        Args:
            query: Search query
            documents: Retrieved documents
            
        Returns:
            Coverage validation result
        """
        doc_contents = "\n\n".join([d.get('content', '') for d in documents])
        
        prompt = f"""Evaluate if these documents adequately cover this query:

Query: {query}

Documents:
{doc_contents[:2000]}  # Limit to 2000 chars

Does the retrieved content provide sufficient information to answer the query?
Rate coverage as: COMPLETE, PARTIAL, or INSUFFICIENT

Response format: COVERAGE: [COMPLETE/PARTIAL/INSUFFICIENT] | REASON: [explanation]"""
        
        try:
            response = self.llm.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=300,
                messages=[{"role": "user", "content": prompt}]
            )
            
            result_text = response.content[0].text
            
            import re
            coverage_match = re.search(r'COVERAGE:\s*(COMPLETE|PARTIAL|INSUFFICIENT)', result_text, re.IGNORECASE)
            reason_match = re.search(r'REASON:\s*(.+)', result_text, re.DOTALL)
            
            coverage = coverage_match.group(1).upper() if coverage_match else "UNKNOWN"
            reason = reason_match.group(1).strip() if reason_match else ""
            
            return {
                'coverage': coverage,
                'is_adequate': coverage in ['COMPLETE', 'PARTIAL'],
                'reason': reason
            }
        
        except Exception as e:
            logger.error(f"Coverage validation failed: {e}")
            return {
                'coverage': 'ERROR',
                'is_adequate': False,
                'reason': str(e)
            }