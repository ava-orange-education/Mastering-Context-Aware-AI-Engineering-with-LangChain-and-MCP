"""
Personally Identifiable Information (PII) detection and handling.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import re
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PIIMatch:
    """Detected PII match"""
    pii_type: str
    value: str
    start: int
    end: int
    confidence: float


class PIIDetector:
    """Detect personally identifiable information in text"""
    
    def __init__(self):
        """Initialize PII detector"""
        self.patterns = self._load_patterns()
    
    def _load_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Load PII detection patterns"""
        return {
            'email': {
                'pattern': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
                'confidence': 0.95
            },
            'phone_us': {
                'pattern': r'\b(\+?1[-.]?)?\(?([0-9]{3})\)?[-.]?([0-9]{3})[-.]?([0-9]{4})\b',
                'confidence': 0.90
            },
            'ssn': {
                'pattern': r'\b\d{3}-\d{2}-\d{4}\b',
                'confidence': 0.95
            },
            'credit_card': {
                'pattern': r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',
                'confidence': 0.85
            },
            'ip_address': {
                'pattern': r'\b(?:\d{1,3}\.){3}\d{1,3}\b',
                'confidence': 0.90
            },
            'date_of_birth': {
                'pattern': r'\b(0?[1-9]|1[0-2])[-/](0?[1-9]|[12][0-9]|3[01])[-/](19|20)\d{2}\b',
                'confidence': 0.80
            },
            'address': {
                'pattern': r'\b\d+\s+[\w\s]+(?:street|st|avenue|ave|road|rd|highway|hwy|square|sq|trail|trl|drive|dr|court|ct|parkway|pkwy|circle|cir|boulevard|blvd)\b',
                'confidence': 0.75
            }
        }
    
    def detect(self, text: str) -> List[PIIMatch]:
        """
        Detect PII in text
        
        Args:
            text: Text to scan
            
        Returns:
            List of detected PII matches
        """
        matches = []
        
        for pii_type, pattern_info in self.patterns.items():
            pattern = pattern_info['pattern']
            confidence = pattern_info['confidence']
            
            for match in re.finditer(pattern, text, re.IGNORECASE):
                pii_match = PIIMatch(
                    pii_type=pii_type,
                    value=match.group(0),
                    start=match.start(),
                    end=match.end(),
                    confidence=confidence
                )
                matches.append(pii_match)
        
        # Sort by position
        matches.sort(key=lambda x: x.start)
        
        return matches
    
    def scan_document(self, text: str, return_summary: bool = True) -> Dict[str, Any]:
        """
        Scan document for PII
        
        Args:
            text: Document text
            return_summary: Whether to return summary stats
            
        Returns:
            Scan results
        """
        matches = self.detect(text)
        
        result = {
            'contains_pii': len(matches) > 0,
            'pii_count': len(matches),
            'matches': [
                {
                    'type': m.pii_type,
                    'value': m.value,
                    'position': {'start': m.start, 'end': m.end},
                    'confidence': m.confidence
                }
                for m in matches
            ]
        }
        
        if return_summary:
            # Summarize by type
            summary = {}
            for match in matches:
                if match.pii_type not in summary:
                    summary[match.pii_type] = 0
                summary[match.pii_type] += 1
            
            result['summary'] = summary
        
        return result
    
    def has_high_risk_pii(self, text: str) -> bool:
        """
        Check if text contains high-risk PII (SSN, credit cards)
        
        Args:
            text: Text to check
            
        Returns:
            True if high-risk PII detected
        """
        high_risk_types = {'ssn', 'credit_card', 'date_of_birth'}
        
        matches = self.detect(text)
        
        return any(m.pii_type in high_risk_types for m in matches)