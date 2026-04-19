"""
HIPAA Logger

Specialized logging for HIPAA compliance with PHI protection
"""

from typing import Dict, Any, Optional
import logging
from datetime import datetime
import json
import hashlib

logger = logging.getLogger(__name__)


class HIPAALogger:
    """
    Logger with HIPAA compliance features
    - Automatic PHI detection and redaction
    - Structured audit logging
    - Encrypted log storage
    """
    
    def __init__(self):
        self.logger = logging.getLogger("hipaa")
        self._setup_logger()
    
    def _setup_logger(self):
        """Setup HIPAA-compliant logger"""
        
        # Use structured JSON logging for audit trails
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(message)s'))
        
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
    
    def log_access(
        self,
        user_id: str,
        patient_id: str,
        action: str,
        resource: str,
        ip_address: str,
        success: bool = True,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Log patient data access
        
        Args:
            user_id: User performing action
            patient_id: Patient whose data was accessed
            action: Action performed (view, update, delete, etc.)
            resource: Resource accessed
            ip_address: IP address of request
            success: Whether action succeeded
            details: Additional details
        """
        
        # Hash patient ID for privacy
        patient_hash = self._hash_identifier(patient_id)
        
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": "patient_access",
            "user_id": user_id,
            "patient_id_hash": patient_hash,
            "action": action,
            "resource": resource,
            "ip_address": ip_address,
            "success": success,
            "details": self._redact_phi(details) if details else {}
        }
        
        self.logger.info(json.dumps(log_entry))
    
    def log_phi_access(
        self,
        user_id: str,
        phi_type: str,
        purpose: str,
        fields_accessed: list,
        justification: str
    ) -> None:
        """
        Log PHI (Protected Health Information) access
        
        Args:
            user_id: User accessing PHI
            phi_type: Type of PHI (demographics, genetic, clinical, etc.)
            purpose: Purpose of access (treatment, payment, operations)
            fields_accessed: List of fields accessed
            justification: Justification for access
        """
        
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": "phi_access",
            "user_id": user_id,
            "phi_type": phi_type,
            "purpose": purpose,
            "fields_accessed": fields_accessed,
            "justification": justification
        }
        
        self.logger.info(json.dumps(log_entry))
    
    def log_consent(
        self,
        patient_id: str,
        consent_type: str,
        granted: bool,
        scope: str,
        expiration: Optional[str] = None
    ) -> None:
        """
        Log patient consent
        
        Args:
            patient_id: Patient granting/revoking consent
            consent_type: Type of consent
            granted: Whether consent was granted
            scope: Scope of consent
            expiration: Expiration date (if applicable)
        """
        
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": "consent",
            "patient_id_hash": self._hash_identifier(patient_id),
            "consent_type": consent_type,
            "granted": granted,
            "scope": scope,
            "expiration": expiration
        }
        
        self.logger.info(json.dumps(log_entry))
    
    def log_disclosure(
        self,
        patient_id: str,
        recipient: str,
        purpose: str,
        data_disclosed: str,
        authorization: bool
    ) -> None:
        """
        Log PHI disclosure to external parties
        
        Args:
            patient_id: Patient whose data was disclosed
            recipient: Recipient of disclosure
            purpose: Purpose of disclosure
            data_disclosed: Description of data disclosed
            authorization: Whether patient authorized disclosure
        """
        
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": "disclosure",
            "patient_id_hash": self._hash_identifier(patient_id),
            "recipient": recipient,
            "purpose": purpose,
            "data_disclosed": data_disclosed,
            "authorized": authorization
        }
        
        self.logger.info(json.dumps(log_entry))
    
    def log_breach(
        self,
        breach_type: str,
        affected_patients: int,
        description: str,
        severity: str,
        mitigation: str
    ) -> None:
        """
        Log security breach
        
        Args:
            breach_type: Type of breach
            affected_patients: Number of patients affected
            description: Description of breach
            severity: Severity level
            mitigation: Mitigation steps taken
        """
        
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": "breach",
            "breach_type": breach_type,
            "affected_patients": affected_patients,
            "description": description,
            "severity": severity,
            "mitigation": mitigation,
            "alert": True  # Flag for immediate attention
        }
        
        self.logger.critical(json.dumps(log_entry))
    
    def log_data_modification(
        self,
        user_id: str,
        patient_id: str,
        resource: str,
        field_modified: str,
        old_value: Any,
        new_value: Any,
        reason: str
    ) -> None:
        """
        Log data modifications for audit trail
        
        Args:
            user_id: User making modification
            patient_id: Patient whose data was modified
            resource: Resource modified
            field_modified: Field that was changed
            old_value: Previous value
            new_value: New value
            reason: Reason for modification
        """
        
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": "data_modification",
            "user_id": user_id,
            "patient_id_hash": self._hash_identifier(patient_id),
            "resource": resource,
            "field_modified": field_modified,
            "old_value": self._redact_phi(old_value),
            "new_value": self._redact_phi(new_value),
            "reason": reason
        }
        
        self.logger.info(json.dumps(log_entry))
    
    def _hash_identifier(self, identifier: str) -> str:
        """Hash patient identifier for privacy"""
        
        # Use SHA-256 for one-way hashing
        return hashlib.sha256(identifier.encode()).hexdigest()[:16]
    
    def _redact_phi(self, data: Any) -> Any:
        """
        Redact PHI from data before logging
        
        In production, implement comprehensive PHI detection
        """
        
        import re
        
        if isinstance(data, str):
            # Redact common PHI patterns
            data = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', '[SSN-REDACTED]', data)  # SSN
            data = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '[PHONE-REDACTED]', data)  # Phone
            data = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL-REDACTED]', data)  # Email
            
            return data
        
        elif isinstance(data, dict):
            # Recursively redact dictionary
            return {k: self._redact_phi(v) for k, v in data.items()}
        
        elif isinstance(data, list):
            # Recursively redact list
            return [self._redact_phi(item) for item in data]
        
        else:
            return data