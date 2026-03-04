"""
Fact checking against source documents.
"""

from typing import Dict, Any, List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FactChecker:
    """Verify facts against source documents"""
    
    def __init__(self, llm_client):
        """
        Initialize fact checker
        
        Args:
            llm_client: LLM client for verification
        """
        self.llm = llm_client
    
    def check_fact(self, claim: str, sources: List[str]) -> Dict[str, Any]:
        """
        Check if claim is supported by sources
        
        Args:
            claim: Factual claim to verify
            sources: Source documents
            
        Returns:
            Verification result
        """
        sources_text = "\n\n".join([f"Source {i+1}:\n{src}" 
                                    for i, src in enumerate(sources)])
        
        prompt = f"""Verify if this claim is supported by the provided sources:

Claim: {claim}

Sources:
{sources_text}

Is the claim supported by the sources?
- Answer YES if the claim is directly supported
- Answer NO if the claim contradicts the sources
- Answer PARTIAL if the claim is partly supported
- Answer UNKNOWN if sources don't address the claim

Response format: VERDICT: [YES/NO/PARTIAL/UNKNOWN] | CONFIDENCE: [0.0-1.0] | EXPLANATION: [brief explanation]"""
        
        try:
            response = self.llm.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=500,
                messages=[{"role": "user", "content": prompt}]
            )
            
            result_text = response.content[0].text
            
            # Parse response
            import re
            verdict_match = re.search(r'VERDICT:\s*(YES|NO|PARTIAL|UNKNOWN)', result_text, re.IGNORECASE)
            confidence_match = re.search(r'CONFIDENCE:\s*(0?\.\d+|1\.0)', result_text)
            explanation_match = re.search(r'EXPLANATION:\s*(.+)', result_text, re.DOTALL)
            
            verdict = verdict_match.group(1).upper() if verdict_match else "UNKNOWN"
            confidence = float(confidence_match.group(1)) if confidence_match else 0.5
            explanation = explanation_match.group(1).strip() if explanation_match else ""
            
            return {
                'claim': claim,
                'supported': verdict == "YES",
                'verdict': verdict,
                'confidence': confidence,
                'explanation': explanation
            }
        
        except Exception as e:
            logger.error(f"Fact checking failed: {e}")
            return {
                'claim': claim,
                'supported': False,
                'verdict': 'ERROR',
                'confidence': 0.0,
                'explanation': f"Error: {str(e)}"
            }
    
    def check_multiple_facts(self, claims: List[str], sources: List[str]) -> List[Dict[str, Any]]:
        """Check multiple claims against sources"""
        results = []
        
        for claim in claims:
            result = self.check_fact(claim, sources)
            results.append(result)
        
        return results
    
    def get_fact_check_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Summarize fact check results"""
        total = len(results)
        supported = sum(1 for r in results if r['supported'])
        
        return {
            'total_claims': total,
            'supported_claims': supported,
            'unsupported_claims': total - supported,
            'support_rate': supported / total if total > 0 else 0,
            'details': results
        }