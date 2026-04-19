"""
Privacy Filter

Filters and redacts PHI from RAG results
"""

from typing import List, Dict, Any
import logging
import re

logger = logging.getLogger(__name__)


class PrivacyFilter:
    """
    Filter to protect patient privacy in RAG results
    """
    
    def __init__(self):
        # PHI patterns to detect and redact
        self.phi_patterns = {
            "ssn": r'\b\d{3}-\d{2}-\d{4}\b',
            "phone": r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
            "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            "date": r'\b\d{1,2}/\d{1,2}/\d{4}\b',
            "mrn": r'\bMRN[:\s]*\d+\b',
            "account_number": r'\b[Aa]ccount[:\s]*\d+\b',
            "ip_address": r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b',
        }
        
        # Common patient identifiers
        self.name_indicators = [
            "patient", "mr.", "mrs.", "ms.", "dr."
        ]
    
    def filter_search_results(
        self,
        results: List[Dict[str, Any]],
        patient_id: str
    ) -> List[Dict[str, Any]]:
        """
        Filter search results to remove PHI
        
        Args:
            results: Search results
            patient_id: Current patient ID (to preserve in context)
        
        Returns:
            Filtered results
        """
        
        filtered_results = []
        
        for result in results:
            # Redact content
            if "content" in result:
                result["content"] = self.redact_phi(
                    result["content"],
                    preserve_patient_id=patient_id
                )
            
            # Redact metadata
            if "metadata" in result:
                result["metadata"] = self._redact_metadata(result["metadata"])
            
            filtered_results.append(result)
        
        return filtered_results
    
    def redact_phi(
        self,
        text: str,
        preserve_patient_id: Optional[str] = None
    ) -> str:
        """
        Redact PHI from text
        
        Args:
            text: Text potentially containing PHI
            preserve_patient_id: Patient ID to preserve (if in context)
        
        Returns:
            Redacted text
        """
        
        redacted_text = text
        
        # Apply PHI pattern redactions
        for phi_type, pattern in self.phi_patterns.items():
            redacted_text = re.sub(
                pattern,
                f'[{phi_type.upper()}-REDACTED]',
                redacted_text
            )
        
        # Detect and redact potential names
        redacted_text = self._redact_names(redacted_text, preserve_patient_id)
        
        return redacted_text
    
    def _redact_names(
        self,
        text: str,
        preserve_patient_id: Optional[str] = None
    ) -> str:
        """
        Redact potential patient names
        
        Args:
            text: Text
            preserve_patient_id: Patient ID to preserve
        
        Returns:
            Text with names redacted
        """
        
        # Simple heuristic: Look for capitalized words after name indicators
        for indicator in self.name_indicators:
            # Pattern: indicator followed by capitalized word(s)
            pattern = r'\b' + indicator + r'\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)'
            
            def replace_name(match):
                if preserve_patient_id and preserve_patient_id in match.group(0):
                    return match.group(0)  # Preserve if it's the current patient
                return f"{indicator} [NAME-REDACTED]"
            
            text = re.sub(pattern, replace_name, text, flags=re.IGNORECASE)
        
        return text
    
    def _redact_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Redact PHI from metadata
        
        Args:
            metadata: Document metadata
        
        Returns:
            Redacted metadata
        """
        
        # Fields that commonly contain PHI
        phi_fields = [
            "patient_name",
            "patient_email",
            "patient_phone",
            "patient_address",
            "author_email",
            "physician_name"
        ]
        
        redacted_metadata = metadata.copy()
        
        for field in phi_fields:
            if field in redacted_metadata:
                redacted_metadata[field] = "[REDACTED]"
        
        # Redact any string values that match PHI patterns
        for key, value in redacted_metadata.items():
            if isinstance(value, str):
                redacted_metadata[key] = self.redact_phi(value)
        
        return redacted_metadata
    
    def validate_no_phi(self, text: str) -> Dict[str, Any]:
        """
        Validate that text contains no PHI
        
        Args:
            text: Text to validate
        
        Returns:
            Validation result with detected PHI
        """
        
        detected_phi = []
        
        for phi_type, pattern in self.phi_patterns.items():
            matches = re.findall(pattern, text)
            if matches:
                detected_phi.append({
                    "type": phi_type,
                    "count": len(matches),
                    "examples": matches[:3]  # First 3 examples
                })
        
        return {
            "phi_free": len(detected_phi) == 0,
            "detected_phi": detected_phi
        }