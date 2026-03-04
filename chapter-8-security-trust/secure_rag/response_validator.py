"""
Validate AI-generated responses before delivery.
"""

from typing import Dict, Any, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ResponseValidator:
    """Validate AI responses for safety and quality"""
    
    def __init__(self, pii_detector, hallucination_detector):
        """
        Initialize response validator
        
        Args:
            pii_detector: PII detector instance
            hallucination_detector: Hallucination detector instance
        """
        self.pii_detector = pii_detector
        self.hallucination_detector = hallucination_detector
    
    def validate(self, response: str, context: Optional[str] = None,
                user_can_view_pii: bool = False) -> Dict[str, Any]:
        """
        Comprehensive response validation
        
        Args:
            response: Generated response
            context: Source context
            user_can_view_pii: Whether user has PII viewing permission
            
        Returns:
            Validation result
        """
        validation_results = {
            'is_valid': True,
            'issues': [],
            'warnings': []
        }
        
        # 1. Check for PII
        pii_scan = self.pii_detector.scan_document(response)
        
        if pii_scan['contains_pii']:
            if not user_can_view_pii:
                validation_results['is_valid'] = False
                validation_results['issues'].append({
                    'type': 'pii_exposure',
                    'severity': 'high',
                    'message': f"Response contains {pii_scan['pii_count']} PII instances",
                    'pii_types': list(pii_scan['summary'].keys())
                })
            else:
                validation_results['warnings'].append({
                    'type': 'pii_present',
                    'message': 'Response contains PII (user authorized)'
                })
        
        # 2. Check for hallucinations
        if context:
            hallucination_check = self.hallucination_detector.check_response(
                response, context
            )
            
            if hallucination_check.is_hallucination:
                if hallucination_check.confidence > 0.8:
                    validation_results['is_valid'] = False
                    validation_results['issues'].append({
                        'type': 'hallucination',
                        'severity': 'high',
                        'message': hallucination_check.reason,
                        'confidence': hallucination_check.confidence
                    })
                else:
                    validation_results['warnings'].append({
                        'type': 'potential_hallucination',
                        'message': hallucination_check.reason,
                        'confidence': hallucination_check.confidence
                    })
        
        # 3. Check response length
        if len(response) > 10000:
            validation_results['warnings'].append({
                'type': 'long_response',
                'message': 'Response exceeds recommended length',
                'length': len(response)
            })
        
        # 4. Check for empty response
        if not response.strip():
            validation_results['is_valid'] = False
            validation_results['issues'].append({
                'type': 'empty_response',
                'severity': 'high',
                'message': 'Response is empty'
            })
        
        return validation_results
    
    def should_block_response(self, validation_result: Dict[str, Any]) -> bool:
        """
        Determine if response should be blocked
        
        Args:
            validation_result: Validation result
            
        Returns:
            True if response should be blocked
        """
        return not validation_result['is_valid']
    
    def get_safe_fallback_response(self) -> str:
        """Get safe fallback response when validation fails"""
        return ("I apologize, but I cannot provide a response to this query. "
                "Please rephrase your question or contact support for assistance.")