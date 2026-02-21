"""
Validation utilities.
"""

from typing import Any, List, Dict
import re


class ValidationUtils:
    """Utilities for input validation"""
    
    @staticmethod
    def is_valid_url(url: str) -> bool:
        """Check if string is valid URL"""
        url_pattern = re.compile(
            r'^https?://'  # http:// or https://
            r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain
            r'localhost|'  # localhost
            r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # IP
            r'(?::\d+)?'  # optional port
            r'(?:/?|[/?]\S+)$', re.IGNORECASE)
        
        return url_pattern.match(url) is not None
    
    @staticmethod
    def is_valid_email(email: str) -> bool:
        """Check if string is valid email"""
        email_pattern = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
        return email_pattern.match(email) is not None
    
    @staticmethod
    def validate_required_fields(data: Dict[str, Any], 
                                required_fields: List[str]) -> Dict[str, Any]:
        """
        Validate required fields in dictionary
        
        Args:
            data: Data dictionary
            required_fields: List of required field names
            
        Returns:
            Validation result
        """
        missing = [field for field in required_fields if field not in data]
        
        if missing:
            return {
                'valid': False,
                'missing_fields': missing
            }
        
        return {'valid': True}
    
    @staticmethod
    def validate_type(value: Any, expected_type: type) -> bool:
        """Check if value is of expected type"""
        return isinstance(value, expected_type)
    
    @staticmethod
    def validate_range(value: float, min_val: float, max_val: float) -> bool:
        """Check if value is within range"""
        return min_val <= value <= max_val