"""
Comprehensive audit logging for compliance and security.
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class AuditLog:
    """Audit log entry"""
    log_id: str
    timestamp: datetime
    user_id: Optional[str]
    action: str
    resource_type: str
    resource_id: Optional[str]
    result: str  # success, failure, denied
    ip_address: Optional[str]
    details: Dict[str, Any] = field(default_factory=dict)


class AuditLogger:
    """Comprehensive audit logging for compliance"""
    
    def __init__(self, log_file: str = "audit.log"):
        """
        Initialize audit logger
        
        Args:
            log_file: Path to audit log file
        """
        self.log_file = log_file
        self.logs: List[AuditLog] = []
        self.log_counter = 0
    
    def log_action(self, action: str, user_id: Optional[str] = None,
                  resource_type: str = "system", resource_id: Optional[str] = None,
                  result: str = "success", ip_address: Optional[str] = None,
                  details: Optional[Dict] = None) -> str:
        """
        Log an action for audit trail
        
        Args:
            action: Action performed
            user_id: User who performed action
            resource_type: Type of resource accessed
            resource_id: Specific resource identifier
            result: Action result (success/failure/denied)
            ip_address: Client IP address
            details: Additional details
            
        Returns:
            Log ID
        """
        self.log_counter += 1
        log_id = f"audit_{self.log_counter}_{int(datetime.now().timestamp())}"
        
        audit_entry = AuditLog(
            log_id=log_id,
            timestamp=datetime.now(),
            user_id=user_id,
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            result=result,
            ip_address=ip_address,
            details=details or {}
        )
        
        self.logs.append(audit_entry)
        
        # Write to file
        self._write_to_file(audit_entry)
        
        return log_id
    
    def _write_to_file(self, audit_entry: AuditLog):
        """Write audit log to file"""
        log_data = {
            'log_id': audit_entry.log_id,
            'timestamp': audit_entry.timestamp.isoformat(),
            'user_id': audit_entry.user_id,
            'action': audit_entry.action,
            'resource_type': audit_entry.resource_type,
            'resource_id': audit_entry.resource_id,
            'result': audit_entry.result,
            'ip_address': audit_entry.ip_address,
            'details': audit_entry.details
        }
        
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(log_data) + '\n')
    
    def query_logs(self, user_id: Optional[str] = None,
                   action: Optional[str] = None,
                   start_time: Optional[datetime] = None,
                   end_time: Optional[datetime] = None) -> List[AuditLog]:
        """
        Query audit logs with filters
        
        Args:
            user_id: Filter by user
            action: Filter by action type
            start_time: Filter by start time
            end_time: Filter by end time
            
        Returns:
            Filtered audit logs
        """
        results = self.logs
        
        if user_id:
            results = [log for log in results if log.user_id == user_id]
        
        if action:
            results = [log for log in results if log.action == action]
        
        if start_time:
            results = [log for log in results if log.timestamp >= start_time]
        
        if end_time:
            results = [log for log in results if log.timestamp <= end_time]
        
        return results
    
    def generate_compliance_report(self, days: int = 30) -> Dict[str, Any]:
        """Generate compliance report for last N days"""
        from datetime import timedelta
        
        cutoff = datetime.now() - timedelta(days=days)
        recent_logs = [log for log in self.logs if log.timestamp >= cutoff]
        
        # Count by action
        action_counts = {}
        for log in recent_logs:
            action_counts[log.action] = action_counts.get(log.action, 0) + 1
        
        # Count by result
        result_counts = {}
        for log in recent_logs:
            result_counts[log.result] = result_counts.get(log.result, 0) + 1
        
        # Count by user
        user_counts = {}
        for log in recent_logs:
            if log.user_id:
                user_counts[log.user_id] = user_counts.get(log.user_id, 0) + 1
        
        return {
            'period_days': days,
            'total_actions': len(recent_logs),
            'by_action': action_counts,
            'by_result': result_counts,
            'by_user': user_counts,
            'generated_at': datetime.now().isoformat()
        }