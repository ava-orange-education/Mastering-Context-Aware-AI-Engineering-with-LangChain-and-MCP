"""
Data anonymization to protect sensitive information.
"""

from typing import Dict, Any, List, Optional
import re
import hashlib
import secrets
from .pii_detector import PIIDetector, PIIMatch


class DataAnonymizer:
    """Anonymize sensitive data in text"""
    
    def __init__(self, pii_detector: Optional[PIIDetector] = None):
        """
        Initialize data anonymizer
        
        Args:
            pii_detector: PII detector instance
        """
        self.pii_detector = pii_detector or PIIDetector()
        self.anonymization_map: Dict[str, str] = {}
    
    def anonymize(self, text: str, strategy: str = "redact") -> Dict[str, Any]:
        """
        Anonymize sensitive data in text
        
        Args:
            text: Text to anonymize
            strategy: Anonymization strategy:
                     - "redact": Replace with [REDACTED]
                     - "mask": Mask with asterisks
                     - "pseudonymize": Replace with consistent fake values
                     - "hash": Replace with hash values
            
        Returns:
            Anonymized text and mapping
        """
        # Detect PII
        matches = self.pii_detector.detect(text)
        
        if not matches:
            return {
                'anonymized_text': text,
                'pii_detected': False,
                'replacements': 0
            }
        
        # Apply anonymization strategy
        anonymized_text = text
        offset = 0  # Track position changes from replacements
        replacements = []
        
        for match in matches:
            original_value = match.value
            
            # Apply strategy
            if strategy == "redact":
                replacement = f"[REDACTED_{match.pii_type.upper()}]"
            elif strategy == "mask":
                replacement = self._mask_value(original_value, match.pii_type)
            elif strategy == "pseudonymize":
                replacement = self._pseudonymize_value(original_value, match.pii_type)
            elif strategy == "hash":
                replacement = self._hash_value(original_value)
            else:
                replacement = "[REDACTED]"
            
            # Replace in text
            start = match.start + offset
            end = match.end + offset
            
            anonymized_text = anonymized_text[:start] + replacement + anonymized_text[end:]
            
            # Update offset
            offset += len(replacement) - len(original_value)
            
            # Record replacement
            replacements.append({
                'original': original_value,
                'replacement': replacement,
                'type': match.pii_type,
                'position': {'start': start, 'end': start + len(replacement)}
            })
        
        return {
            'anonymized_text': anonymized_text,
            'pii_detected': True,
            'replacements': len(replacements),
            'replacement_details': replacements
        }
    
    def _mask_value(self, value: str, pii_type: str) -> str:
        """Mask value with asterisks"""
        if pii_type == 'email':
            # Mask email: keep first char and domain
            parts = value.split('@')
            if len(parts) == 2:
                username = parts[0]
                domain = parts[1]
                masked_username = username[0] + '*' * (len(username) - 1)
                return f"{masked_username}@{domain}"
        
        elif pii_type == 'phone_us':
            # Mask phone: show last 4 digits
            digits = re.sub(r'\D', '', value)
            if len(digits) >= 4:
                return '*' * (len(digits) - 4) + digits[-4:]
        
        elif pii_type == 'ssn':
            # Mask SSN: show last 4 digits
            return '***-**-' + value[-4:]
        
        elif pii_type == 'credit_card':
            # Mask credit card: show last 4 digits
            digits = re.sub(r'\D', '', value)
            if len(digits) >= 4:
                return '*' * (len(digits) - 4) + digits[-4:]
        
        # Default masking
        if len(value) <= 4:
            return '*' * len(value)
        return value[:2] + '*' * (len(value) - 4) + value[-2:]
    
    def _pseudonymize_value(self, value: str, pii_type: str) -> str:
        """Replace with consistent pseudonym"""
        # Check if we've seen this value before
        if value in self.anonymization_map:
            return self.anonymization_map[value]
        
        # Generate pseudonym based on type
        if pii_type == 'email':
            pseudonym = f"user{len(self.anonymization_map) + 1}@example.com"
        elif pii_type == 'phone_us':
            pseudonym = f"555-0{100 + len(self.anonymization_map):03d}"
        elif pii_type == 'ssn':
            pseudonym = f"000-00-{len(self.anonymization_map):04d}"
        elif pii_type == 'credit_card':
            pseudonym = f"0000-0000-0000-{len(self.anonymization_map):04d}"
        elif pii_type == 'address':
            pseudonym = f"{100 + len(self.anonymization_map)} Main Street"
        else:
            pseudonym = f"[PSEUDONYM_{len(self.anonymization_map)}]"
        
        # Store mapping
        self.anonymization_map[value] = pseudonym
        
        return pseudonym
    
    def _hash_value(self, value: str) -> str:
        """Replace with hash"""
        hash_value = hashlib.sha256(value.encode()).hexdigest()[:16]
        return f"[HASH:{hash_value}]"
    
    def deanonymize(self, anonymized_text: str) -> Optional[str]:
        """
        Attempt to reverse pseudonymization
        
        Args:
            anonymized_text: Anonymized text
            
        Returns:
            Original text if mapping exists, None otherwise
        """
        if not self.anonymization_map:
            return None
        
        # Create reverse mapping
        reverse_map = {v: k for k, v in self.anonymization_map.items()}
        
        # Replace pseudonyms with original values
        result = anonymized_text
        for pseudonym, original in reverse_map.items():
            result = result.replace(pseudonym, original)
        
        return result
    
    def clear_mapping(self):
        """Clear anonymization mapping"""
        self.anonymization_map.clear()