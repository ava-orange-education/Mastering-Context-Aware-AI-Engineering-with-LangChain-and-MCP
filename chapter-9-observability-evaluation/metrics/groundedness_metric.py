"""
Groundedness metric to measure how well responses are grounded in source material.
"""

from typing import Dict, Any, List, Optional
import re
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GroundednessMetric:
    """Measure how well responses are grounded in retrieved context"""
    
    def __init__(self, llm_client=None):
        """
        Initialize groundedness metric
        
        Args:
            llm_client: Optional LLM client for advanced checking
        """
        self.llm = llm_client
    
    def calculate(self, response: str, context: str) -> Dict[str, Any]:
        """
        Calculate groundedness score
        
        Args:
            response: Generated response
            context: Source context/documents
            
        Returns:
            Groundedness metrics
        """
        # Extract claims from response
        claims = self._extract_claims(response)
        
        if not claims:
            return {
                'groundedness_score': 1.0,
                'grounded_claims': 0,
                'total_claims': 0,
                'ungrounded_claims': [],
                'method': 'keyword_overlap'
            }
        
        # Check each claim against context
        grounded_count = 0
        ungrounded = []
        
        for claim in claims:
            if self._is_claim_grounded(claim, context):
                grounded_count += 1
            else:
                ungrounded.append(claim)
        
        groundedness_score = grounded_count / len(claims) if claims else 1.0
        
        return {
            'groundedness_score': groundedness_score,
            'grounded_claims': grounded_count,
            'total_claims': len(claims),
            'ungrounded_claims': ungrounded,
            'method': 'keyword_overlap'
        }
    
    def _extract_claims(self, text: str) -> List[str]:
        """
        Extract factual claims from text
        
        Args:
            text: Text to analyze
            
        Returns:
            List of claims
        """
        # Split into sentences
        sentences = re.split(r'[.!?]+', text)
        
        claims = []
        for sentence in sentences:
            sentence = sentence.strip()
            
            # Filter for sentences that look like factual claims
            if sentence and len(sentence) > 10:
                # Contains numbers, names, or specific assertions
                if (re.search(r'\d+', sentence) or  # Numbers
                    re.search(r'\b[A-Z][a-z]+ [A-Z][a-z]+\b', sentence) or  # Names
                    re.search(r'\b(is|are|was|were|has|have|will|increased|decreased|shows)\b', sentence)):
                    claims.append(sentence)
        
        return claims
    
    def _is_claim_grounded(self, claim: str, context: str) -> bool:
        """
        Check if claim is grounded in context using keyword overlap
        
        Args:
            claim: Claim to check
            context: Source context
            
        Returns:
            True if claim appears grounded
        """
        # Normalize text
        claim_lower = claim.lower()
        context_lower = context.lower()
        
        # Extract content words (filter out stop words)
        stop_words = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'has', 'have', 
            'had', 'be', 'been', 'being', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'this', 'that', 'these', 'those'
        }
        
        claim_words = set(claim_lower.split()) - stop_words
        context_words = set(context_lower.split()) - stop_words
        
        if not claim_words:
            return True  # Empty claim considered grounded
        
        # Calculate overlap
        overlap = len(claim_words & context_words)
        overlap_ratio = overlap / len(claim_words)
        
        # Require at least 60% overlap
        return overlap_ratio >= 0.6
    
    def calculate_with_llm(self, response: str, context: str) -> Dict[str, Any]:
        """
        Calculate groundedness using LLM for more sophisticated checking
        
        Args:
            response: Generated response
            context: Source context
            
        Returns:
            Groundedness metrics
        """
        if not self.llm:
            return self.calculate(response, context)
        
        prompt = f"""Evaluate how well this response is grounded in the provided context.

Response:
{response}

Context:
{context}

For each claim in the response, determine if it is supported by the context.

Provide your assessment in this format:
GROUNDED_CLAIMS: [number]
TOTAL_CLAIMS: [number]
GROUNDEDNESS_SCORE: [0.0-1.0]
UNGROUNDED: [list any ungrounded claims]"""
        
        try:
            result = self.llm.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=1000,
                messages=[{"role": "user", "content": prompt}]
            )
            
            result_text = result.content[0].text
            
            # Parse result
            grounded_match = re.search(r'GROUNDED_CLAIMS:\s*(\d+)', result_text)
            total_match = re.search(r'TOTAL_CLAIMS:\s*(\d+)', result_text)
            score_match = re.search(r'GROUNDEDNESS_SCORE:\s*(0?\.\d+|1\.0)', result_text)
            
            grounded = int(grounded_match.group(1)) if grounded_match else 0
            total = int(total_match.group(1)) if total_match else 0
            score = float(score_match.group(1)) if score_match else 0.0
            
            return {
                'groundedness_score': score,
                'grounded_claims': grounded,
                'total_claims': total,
                'method': 'llm_verification'
            }
        
        except Exception as e:
            logger.error(f"LLM groundedness check failed: {e}")
            return self.calculate(response, context)