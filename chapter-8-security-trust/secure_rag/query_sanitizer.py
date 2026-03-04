"""
Sanitize and validate user queries for security.
"""

from typing import Dict, Any, List
import re
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QuerySanitizer:
    """Sanitize queries to prevent injection attacks"""
    
    def __init__(self):
        """Initialize query sanitizer"""
        self.sql_patterns = [
            r'\b(DROP|DELETE|INSERT|UPDATE|ALTER|TRUNCATE)\b',
            r'--',
            r';',
            r'\bUNION\b.*\bSELECT\b',
            r'\bEXEC\b',
            r'\bEXECUTE\b'
        ]
        
        self.script_patterns = [
            r'<script',
            r'javascript:',
            r'onerror=',
            r'onclick='
        ]
    
    def sanitize(self, query: str) -> Dict[str, Any]:
        """
        Sanitize query and check for malicious patterns
        
        Args:
            query: User query
            
        Returns:
            Sanitization result
        """
        original_query = query
        issues = []
        
        # Check for SQL injection patterns
        for pattern in self.sql_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                issues.append({
                    'type': 'sql_injection',
                    'pattern': pattern,
                    'severity': 'high'
                })
        
        # Check for script injection
        for pattern in self.script_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                issues.append({
                    'type': 'script_injection',
                    'pattern': pattern,
                    'severity': 'high'
                })
        
        # Check length
        if len(query) > 5000:
            issues.append({
                'type': 'excessive_length',
                'length': len(query),
                'severity': 'medium'
            })
        
        # Basic sanitization - remove potential harmful characters
        sanitized = query
        # Remove null bytes
        sanitized = sanitized.replace('\x00', '')
        # Limit to printable characters
        sanitized = ''.join(c for c in sanitized if c.isprintable() or c.isspace())
        
        return {
            'original': original_query,
            'sanitized': sanitized,
            'is_safe': len(issues) == 0,
            'issues': issues,
            'modified': sanitized != original_query
        }
    
    def validate_query_length(self, query: str, max_length: int = 5000) -> bool:
        """Check if query length is acceptable"""
        return len(query) <= max_length
    
    def detect_prompt_injection(self, query: str) -> Dict[str, Any]:
        """
        Detect prompt injection attempts
        
        Args:
            query: User query
            
        Returns:
            Detection result
        """
        injection_patterns = [
            r'ignore\s+(previous|above|prior)\s+instructions',
            r'disregard\s+(previous|above|prior)',
            r'forget\s+(everything|all|previous)',
            r'new\s+instructions:',
            r'system:',
            r'<\|im_start\|>',
            r'<\|im_end\|>'
        ]
        
        detected = []
        for pattern in injection_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                detected.append(pattern)
        
        return {
            'injection_detected': len(detected) > 0,
            'patterns_found': detected,
            'risk_level': 'high' if detected else 'low'
        }