"""
Relevance metric to measure how well responses address the query.
"""

from typing import Dict, Any, List
import re
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RelevanceMetric:
    """Measure response relevance to query"""
    
    def __init__(self, llm_client=None):
        """
        Initialize relevance metric
        
        Args:
            llm_client: Optional LLM client for evaluation
        """
        self.llm = llm_client
    
    def calculate(self, query: str, response: str) -> Dict[str, Any]:
        """
        Calculate relevance score
        
        Args:
            query: User query
            response: Generated response
            
        Returns:
            Relevance metrics
        """
        # Keyword overlap approach
        keyword_score = self._keyword_overlap(query, response)
        
        # Intent matching
        intent_score = self._check_intent_match(query, response)
        
        # Overall relevance
        relevance_score = (keyword_score + intent_score) / 2
        
        return {
            'relevance_score': relevance_score,
            'keyword_overlap_score': keyword_score,
            'intent_match_score': intent_score,
            'method': 'keyword_and_intent'
        }
    
    def _keyword_overlap(self, query: str, response: str) -> float:
        """
        Calculate keyword overlap between query and response
        
        Args:
            query: User query
            response: Generated response
            
        Returns:
            Overlap score (0-1)
        """
        # Extract keywords (remove stop words)
        stop_words = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'has', 'have',
            'what', 'when', 'where', 'who', 'why', 'how', 'do', 'does',
            'can', 'could', 'should', 'would', 'will', 'in', 'on', 'at'
        }
        
        query_words = set(query.lower().split()) - stop_words
        response_words = set(response.lower().split()) - stop_words
        
        if not query_words:
            return 1.0
        
        # Calculate overlap
        overlap = len(query_words & response_words)
        overlap_ratio = overlap / len(query_words)
        
        return min(1.0, overlap_ratio)
    
    def _check_intent_match(self, query: str, response: str) -> float:
        """
        Check if response matches query intent
        
        Args:
            query: User query
            response: Generated response
            
        Returns:
            Intent match score (0-1)
        """
        query_lower = query.lower()
        response_lower = response.lower()
        
        # Identify query type
        if any(word in query_lower for word in ['what', 'define', 'meaning']):
            # Definition query - expect explanation
            if len(response.split()) >= 10:  # Has explanation
                return 1.0
            return 0.5
        
        elif any(word in query_lower for word in ['how many', 'count', 'number']):
            # Counting query - expect number
            if re.search(r'\d+', response):
                return 1.0
            return 0.3
        
        elif any(word in query_lower for word in ['list', 'name', 'enumerate']):
            # List query - expect list/enumeration
            if re.search(r'\n\s*[-•\d]', response) or any(word in response_lower for word in ['first', 'second', 'third']):
                return 1.0
            return 0.5
        
        elif any(word in query_lower for word in ['why', 'explain', 'reason']):
            # Explanation query - expect detailed explanation
            if len(response.split()) >= 20:
                return 1.0
            return 0.6
        
        elif any(word in query_lower for word in ['compare', 'difference', 'versus', 'vs']):
            # Comparison query - expect comparison
            if any(word in response_lower for word in ['while', 'whereas', 'however', 'contrast', 'different']):
                return 1.0
            return 0.5
        
        # Default - general query
        return 0.7
    
    def calculate_with_llm(self, query: str, response: str) -> Dict[str, Any]:
        """
        Calculate relevance using LLM evaluation
        
        Args:
            query: User query
            response: Generated response
            
        Returns:
            Relevance metrics
        """
        if not self.llm:
            return self.calculate(query, response)
        
        prompt = f"""Evaluate how well this response addresses the query.

Query: {query}

Response: {response}

Rate the relevance on a scale of 0-1:
- 1.0: Directly and completely answers the query
- 0.7-0.9: Mostly answers the query with minor gaps
- 0.4-0.6: Partially answers the query
- 0.0-0.3: Does not answer the query

Provide: RELEVANCE_SCORE: [0.0-1.0]
Explanation: [brief reasoning]"""
        
        try:
            result = self.llm.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=500,
                messages=[{"role": "user", "content": prompt}]
            )
            
            result_text = result.content[0].text
            
            score_match = re.search(r'RELEVANCE_SCORE:\s*(0?\.\d+|1\.0)', result_text)
            score = float(score_match.group(1)) if score_match else 0.5
            
            return {
                'relevance_score': score,
                'method': 'llm_evaluation',
                'explanation': result_text
            }
        
        except Exception as e:
            logger.error(f"LLM relevance check failed: {e}")
            return self.calculate(query, response)