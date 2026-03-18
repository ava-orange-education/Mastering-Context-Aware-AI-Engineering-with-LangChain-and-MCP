"""
Factuality metric to measure accuracy of factual claims.
"""

from typing import Dict, Any, List, Optional
import re
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FactualityMetric:
    """Measure factual accuracy of responses"""
    
    def __init__(self, llm_client=None):
        """
        Initialize factuality metric
        
        Args:
            llm_client: LLM client for fact checking
        """
        self.llm = llm_client
    
    def calculate(self, response: str, ground_truth: Optional[str] = None,
                 sources: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Calculate factuality score
        
        Args:
            response: Generated response
            ground_truth: Known correct answer (if available)
            sources: Source documents for verification
            
        Returns:
            Factuality metrics
        """
        if ground_truth:
            return self._check_against_ground_truth(response, ground_truth)
        elif sources:
            return self._check_against_sources(response, sources)
        else:
            return self._check_hallucination_indicators(response)
    
    def _check_against_ground_truth(self, response: str, ground_truth: str) -> Dict[str, Any]:
        """Check response against known correct answer"""
        
        # Simple token overlap for basic checking
        response_words = set(response.lower().split())
        truth_words = set(ground_truth.lower().split())
        
        if not truth_words:
            return {'factuality_score': 0.0, 'method': 'ground_truth_comparison'}
        
        # Calculate overlap
        overlap = len(response_words & truth_words)
        precision = overlap / len(response_words) if response_words else 0
        recall = overlap / len(truth_words) if truth_words else 0
        
        # F1 score
        if precision + recall > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0.0
        
        return {
            'factuality_score': f1,
            'precision': precision,
            'recall': recall,
            'method': 'ground_truth_comparison'
        }
    
    def _check_against_sources(self, response: str, sources: List[str]) -> Dict[str, Any]:
        """Check response claims against source documents"""
        
        # Extract claims
        claims = self._extract_factual_claims(response)
        
        if not claims:
            return {
                'factuality_score': 1.0,
                'verified_claims': 0,
                'total_claims': 0,
                'method': 'source_verification'
            }
        
        # Check each claim
        verified_count = 0
        for claim in claims:
            if self._is_claim_in_sources(claim, sources):
                verified_count += 1
        
        factuality_score = verified_count / len(claims)
        
        return {
            'factuality_score': factuality_score,
            'verified_claims': verified_count,
            'total_claims': len(claims),
            'method': 'source_verification'
        }
    
    def _check_hallucination_indicators(self, response: str) -> Dict[str, Any]:
        """Check for common hallucination indicators"""
        
        # Patterns that often indicate hallucinations
        hallucination_patterns = [
            r'\b(definitely|certainly|absolutely|guaranteed)\b',  # Overconfidence
            r'\b(studies show|research indicates|experts say)\b(?!\s+\[)',  # Vague sources
            r'\$\d+(\.\d{2})?\s+(million|billion)',  # Specific numbers without citation
            r'\b\d{1,2}/\d{1,2}/\d{2,4}\b',  # Specific dates
        ]
        
        indicator_count = 0
        for pattern in hallucination_patterns:
            if re.search(pattern, response, re.IGNORECASE):
                indicator_count += 1
        
        # Score inversely related to indicators
        max_indicators = len(hallucination_patterns)
        score = 1.0 - (indicator_count / max_indicators)
        
        return {
            'factuality_score': max(0.0, score),
            'hallucination_indicators': indicator_count,
            'method': 'hallucination_detection'
        }
    
    def _extract_factual_claims(self, text: str) -> List[str]:
        """Extract factual claims from text"""
        sentences = re.split(r'[.!?]+', text)
        
        claims = []
        for sentence in sentences:
            sentence = sentence.strip()
            # Look for factual statements
            if sentence and (
                re.search(r'\d+', sentence) or
                re.search(r'\b(is|are|was|were|has|have)\b', sentence)
            ):
                claims.append(sentence)
        
        return claims
    
    def _is_claim_in_sources(self, claim: str, sources: List[str]) -> bool:
        """Check if claim appears in sources"""
        claim_lower = claim.lower()
        
        for source in sources:
            source_lower = source.lower()
            
            # Simple substring check
            if claim_lower in source_lower:
                return True
            
            # Check keyword overlap
            claim_words = set(claim_lower.split())
            source_words = set(source_lower.split())
            
            overlap = len(claim_words & source_words)
            if overlap / len(claim_words) >= 0.7:
                return True
        
        return False
    
    def calculate_with_llm(self, response: str, sources: List[str]) -> Dict[str, Any]:
        """
        Calculate factuality using LLM
        
        Args:
            response: Generated response
            sources: Source documents
            
        Returns:
            Factuality metrics
        """
        if not self.llm:
            return self._check_against_sources(response, sources)
        
        sources_text = "\n\n".join([f"Source {i+1}: {s}" for i, s in enumerate(sources)])
        
        prompt = f"""Verify the factual accuracy of this response against the provided sources.

Response:
{response}

Sources:
{sources_text}

For each factual claim in the response:
1. Identify if it's supported by the sources
2. Mark as VERIFIED or UNVERIFIED

Provide:
VERIFIED_CLAIMS: [count]
TOTAL_CLAIMS: [count]
FACTUALITY_SCORE: [0.0-1.0]"""
        
        try:
            result = self.llm.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=1000,
                messages=[{"role": "user", "content": prompt}]
            )
            
            result_text = result.content[0].text
            
            verified_match = re.search(r'VERIFIED_CLAIMS:\s*(\d+)', result_text)
            total_match = re.search(r'TOTAL_CLAIMS:\s*(\d+)', result_text)
            score_match = re.search(r'FACTUALITY_SCORE:\s*(0?\.\d+|1\.0)', result_text)
            
            verified = int(verified_match.group(1)) if verified_match else 0
            total = int(total_match.group(1)) if total_match else 0
            score = float(score_match.group(1)) if score_match else 0.0
            
            return {
                'factuality_score': score,
                'verified_claims': verified,
                'total_claims': total,
                'method': 'llm_verification'
            }
        
        except Exception as e:
            logger.error(f"LLM factuality check failed: {e}")
            return self._check_against_sources(response, sources)