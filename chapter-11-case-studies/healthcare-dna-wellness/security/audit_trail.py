"""
Audit Trail

Comprehensive audit logging for HIPAA compliance
"""

from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import logging
import uuid
import sys
sys.path.append('../..')

from shared.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class AuditLogger:
    """
    Audit trail logger for HIPAA compliance
    """
    
    def __init__(self):
        # In production, use database for audit logs
        # For demo, use in-memory storage
        self.audit_log: List[Dict[str, Any]] = []
    
    async def log_access(
        self,
        user_id: str,
        patient_id: str,
        action: str,
        ip_address: str,
        data_fields: Optional[List[str]] = None,
        timestamp: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Log patient data access
        
        Args:
            user_id: User accessing data
            patient_id: Patient whose data was accessed
            action: Action performed
            ip_address: IP address of request
            data_fields: Fields accessed
            timestamp: Timestamp (defaults to now)
        
        Returns:
            Audit log entry
        """
        
        entry = {
            "id": str(uuid.uuid4()),
            "timestamp": (timestamp or datetime.utcnow()).isoformat(),
            "event_type": "patient_access",
            "user_id": user_id,
            "patient_id": patient_id,
            "action": action,
            "ip_address": ip_address,
            "data_fields": data_fields or [],
            "success": True
        }
        
        self.audit_log.append(entry)
        
        logger.info(
            f"AUDIT: {user_id} accessed {patient_id} ({action}) "
            f"from {ip_address}"
        )
        
        return entry
    
    async def log_modification(
        self,
        user_id: str,
        patient_id: str,
        resource: str,
        field: str,
        old_value: Any,
        new_value: Any,
        reason: str
    ) -> Dict[str, Any]:
        """
        Log data modification
        
        Args:
            user_id: User making modification
            patient_id: Patient whose data was modified
            resource: Resource modified
            field: Field modified
            old_value: Previous value
            new_value: New value
            reason: Reason for modification
        
        Returns:
            Audit log entry
        """
        
        entry = {
            "id": str(uuid.uuid4()),
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": "data_modification",
            "user_id": user_id,
            "patient_id": patient_id,
            "resource": resource,
            "field": field,
            "old_value": str(old_value),
            "new_value": str(new_value),
            "reason": reason,
            "success": True
        }
        
        self.audit_log.append(entry)
        
        logger.info(
            f"AUDIT: {user_id} modified {patient_id}/{resource}/{field}"
        )
        
        return entry
    
    async def get_patient_access_log(
        self,
        patient_id: str,
        days: int = 30
    ) -> List[Dict[str, Any]]:
        """
        Get access log for patient
        
        Args:
            patient_id: Patient ID
            days: Number of days to retrieve
        
        Returns:
            List of audit entries
        """
        
        cutoff = datetime.utcnow() - timedelta(days=days)
        
        entries = [
            entry for entry in self.audit_log
            if entry.get("patient_id") == patient_id
            and datetime.fromisoformat(entry["timestamp"]) > cutoff
        ]
        
        # Sort by timestamp descending
        entries.sort(key=lambda x: x["timestamp"], reverse=True)
        
        return entries
    
    async def get_user_activity(
        self,
        user_id: str,
        days: int = 30
    ) -> List[Dict[str, Any]]:
        """
        Get activity log for user
        
        Args:
            user_id: User ID
            days: Number of days to retrieve
        
        Returns:
            List of audit entries
        """
        
        cutoff = datetime.utcnow() - timedelta(days=days)
        
        entries = [
            entry for entry in self.audit_log
            if entry.get("user_id") == user_id
            and datetime.fromisoformat(entry["timestamp"]) > cutoff
        ]
        
        entries.sort(key=lambda x: x["timestamp"], reverse=True)
        
        return entries
    
    async def search_audit_log(
        self,
        filters: Dict[str, Any],
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Search audit log with filters
        
        Args:
            filters: Search filters
            limit: Maximum results
        
        Returns:
            Matching audit entries
        """
        
        results = []
        
        for entry in self.audit_log:
            # Check if entry matches all filters
            matches = all(
                entry.get(key) == value
                for key, value in filters.items()
            )
            
            if matches:
                results.append(entry)
                
                if len(results) >= limit:
                    break
        
        return results