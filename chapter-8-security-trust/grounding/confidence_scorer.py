"""
Confidence scoring for AI-generated responses.
"""

from typing import Dict, Any, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ConfidenceScorer:
    """Score confidence in AI-generated responses"""
    
    def __init__(self):
        """Initialize confidence scorer"""
        self.hedging_words = [
            'might', 'maybe', 'perhaps', 'possibly', 'likely',
            'probably', 'seems', 'appears', 'suggests', 'indicates'
        ]
        
        self.confident_words = [
            'definitely', 'certainly', 'absolutely', 'guaranteed',
            'always', 'never', 'must', 'will'
        ]
    
    def score_response(self, response: str, citations_count: int = 0,
                      grounding_score: float = 0.0) -> Dict[str, Any]:
        """
        Calculate confidence score for response
        
        Args:
            response: Response text
            citations_count: Number of citations
            grounding_score: Grounding score (0-1)
            
        Returns:
            Confidence assessment
        """
        # Count hedging vs confident language
        words = response.lower().split()
        hedging_count = sum(1 for word in words if word in self.hedging_words)
        confident_count = sum(1 for word in words if word in self.confident_words)
        
        # Base score from grounding
        base_score = grounding_score
        
        # Adjust for citations (more citations = higher confidence)
        citation_bonus = min(citations_count * 0.05, 0.2)  # Max 0.2 bonus
        
        # Adjust for language (hedging increases confidence, overconfidence decreases)
        language_factor = 0.0
        if hedging_count > 0:
            language_factor += 0.1
        if confident_count > 2:
            language_factor -= 0.1
        
        # Calculate final score
        confidence_score = base_score + citation_bonus + language_factor
        confidence_score = max(0.0, min(1.0, confidence_score))  # Clamp to 0-1
        
        # Determine confidence level
        if confidence_score >= 0.8:
            confidence_level = "high"
        elif confidence_score >= 0.5:
            confidence_level = "medium"
        else:
            confidence_level = "low"
        
        return {
            'confidence_score': confidence_score,
            'confidence_level': confidence_level,
            'grounding_score': grounding_score,
            'citations_count': citations_count,
            'hedging_count': hedging_count,
            'overconfident_count': confident_count,
            'factors': {
                'base': base_score,
                'citations': citation_bonus,
                'language': language_factor
            }
        }