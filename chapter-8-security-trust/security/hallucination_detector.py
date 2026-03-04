"""
Hallucination detection to identify potentially false or unsupported claims.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import logging
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class HallucinationCheck:
    """Result of hallucination detection"""
    is_hallucination: bool
    confidence: float
    reason: str
    flagged_content: Optional[str] = None
    severity: str = "low"  # low, medium, high


class HallucinationDetector:
    """Detects potential hallucinations in AI-generated content"""
    
    def __init__(self, llm_client):
        """
        Initialize hallucination detector
        
        Args:
            llm_client: LLM client for analysis
        """
        self.llm = llm_client
        self.patterns = self._load_patterns()
    
    def _load_patterns(self) -> List[Dict[str, str]]:
        """Load known hallucination patterns"""
        return [
            {
                'pattern': r'\b(definitely|certainly|absolutely|guaranteed)\b',
                'reason': 'Overconfident claims without hedging',
                'severity': 'medium'
            },
            {
                'pattern': r'\b(studies show|research indicates|experts say)\b(?! \[)',
                'reason': 'Vague source references without citations',
                'severity': 'high'
            },
            {
                'pattern': r'\b\d{1,2}/\d{1,2}/\d{2,4}\b',
                'reason': 'Specific dates that may be fabricated',
                'severity': 'medium'
            },
            {
                'pattern': r'\$\d+(\.\d{2})? (million|billion)',
                'reason': 'Specific financial figures without source',
                'severity': 'high'
            }
        ]
    
    def check_response(self, response: str, context: Optional[str] = None) -> HallucinationCheck:
        """
        Check response for potential hallucinations
        
        Args:
            response: Generated response text
            context: Retrieved context that response should be grounded in
            
        Returns:
            HallucinationCheck result
        """
        # Pattern-based detection
        for pattern_info in self.patterns:
            matches = re.findall(pattern_info['pattern'], response, re.IGNORECASE)
            if matches:
                return HallucinationCheck(
                    is_hallucination=True,
                    confidence=0.7,
                    reason=pattern_info['reason'],
                    flagged_content=matches[0] if matches else None,
                    severity=pattern_info['severity']
                )
        
        # Context grounding check
        if context:
            grounding_check = self._check_grounding(response, context)
            if not grounding_check['is_grounded']:
                return HallucinationCheck(
                    is_hallucination=True,
                    confidence=grounding_check['confidence'],
                    reason="Response contains claims not supported by context",
                    severity='high'
                )
        
        # LLM-based verification
        llm_check = self._llm_verification(response, context)
        
        return llm_check
    
    def _check_grounding(self, response: str, context: str) -> Dict[str, Any]:
        """Check if response is grounded in provided context"""
        # Extract key claims from response
        claims = self._extract_claims(response)
        
        # Check each claim against context
        ungrounded_claims = []
        for claim in claims:
            if not self._claim_in_context(claim, context):
                ungrounded_claims.append(claim)
        
        grounding_score = 1.0 - (len(ungrounded_claims) / max(len(claims), 1))
        
        return {
            'is_grounded': grounding_score >= 0.8,
            'confidence': grounding_score,
            'ungrounded_claims': ungrounded_claims
        }
    
    def _extract_claims(self, text: str) -> List[str]:
        """Extract factual claims from text"""
        # Split into sentences
        sentences = re.split(r'[.!?]+', text)
        
        # Filter for factual claims (containing numbers, names, specific details)
        claims = []
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and (
                re.search(r'\d+', sentence) or  # Contains numbers
                re.search(r'\b[A-Z][a-z]+ [A-Z][a-z]+\b', sentence) or  # Contains names
                re.search(r'\b(is|are|was|were|has|have)\b', sentence)  # Factual statements
            ):
                claims.append(sentence)
        
        return claims
    
    def _claim_in_context(self, claim: str, context: str) -> bool:
        """Check if claim is supported by context"""
        # Simple keyword overlap check
        claim_words = set(re.findall(r'\w+', claim.lower()))
        context_words = set(re.findall(r'\w+', context.lower()))
        
        # Remove common words
        stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'has', 'have', 'had'}
        claim_words = claim_words - stop_words
        context_words = context_words - stop_words
        
        # Calculate overlap
        overlap = len(claim_words & context_words) / max(len(claim_words), 1)
        
        return overlap >= 0.6  # 60% keyword overlap threshold
    
    def _llm_verification(self, response: str, context: Optional[str]) -> HallucinationCheck:
        """Use LLM to verify response accuracy"""
        prompt = f"""Analyze this AI-generated response for potential hallucinations or false information:

Response: {response}

{f"Context it should be grounded in: {context}" if context else ""}

Check for:
1. Unsupported factual claims
2. Invented sources or citations
3. Fabricated statistics or numbers
4. Made-up names, dates, or events

Is this response likely to contain hallucinations? Respond with:
- "YES" if hallucinations are likely
- "NO" if response appears accurate
- Confidence score (0.0-1.0)
- Brief explanation

Format: VERDICT: YES/NO | CONFIDENCE: 0.XX | REASON: explanation"""
        
        try:
            verification = self.llm.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=300,
                messages=[{"role": "user", "content": prompt}]
            )
            
            result_text = verification.content[0].text
            
            # Parse result
            verdict_match = re.search(r'VERDICT:\s*(YES|NO)', result_text, re.IGNORECASE)
            confidence_match = re.search(r'CONFIDENCE:\s*(0?\.\d+|1\.0)', result_text)
            reason_match = re.search(r'REASON:\s*(.+)', result_text, re.DOTALL)
            
            is_hallucination = verdict_match.group(1).upper() == 'YES' if verdict_match else False
            confidence = float(confidence_match.group(1)) if confidence_match else 0.5
            reason = reason_match.group(1).strip() if reason_match else "LLM verification inconclusive"
            
            return HallucinationCheck(
                is_hallucination=is_hallucination,
                confidence=confidence,
                reason=reason,
                severity='high' if is_hallucination and confidence > 0.8 else 'medium'
            )
        
        except Exception as e:
            logger.error(f"LLM verification failed: {e}")
            return HallucinationCheck(
                is_hallucination=False,
                confidence=0.0,
                reason=f"Verification error: {str(e)}",
                severity='low'
            )
    
    def batch_check(self, responses: List[str], contexts: Optional[List[str]] = None) -> List[HallucinationCheck]:
        """Check multiple responses for hallucinations"""
        if contexts is None:
            contexts = [None] * len(responses)
        
        results = []
        for response, context in zip(responses, contexts):
            result = self.check_response(response, context)
            results.append(result)
        
        return results