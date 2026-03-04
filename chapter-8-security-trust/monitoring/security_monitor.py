"""
Real-time security monitoring and threat detection.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SecurityEvent:
    """Security event"""
    event_id: str
    event_type: str
    severity: str  # "info", "warning", "critical"
    description: str
    user_id: Optional[str] = None
    agent_id: Optional[str] = None
    ip_address: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SecurityAlert:
    """Security alert"""
    alert_id: str
    alert_type: str
    severity: str
    message: str
    events: List[SecurityEvent]
    triggered_at: datetime
    acknowledged: bool = False


class SecurityMonitor:
    """Monitor security events and detect threats"""
    
    def __init__(self):
        """Initialize security monitor"""
        self.events: List[SecurityEvent] = []
        self.alerts: List[SecurityAlert] = []
        self.event_counter = 0
        self.alert_counter = 0
        
        # Track metrics for anomaly detection
        self.failed_auth_attempts: Dict[str, List[datetime]] = defaultdict(list)
        self.query_counts: Dict[str, List[datetime]] = defaultdict(list)
        self.pii_access_attempts: Dict[str, List[datetime]] = defaultdict(list)
    
    def log_event(self, event_type: str, severity: str, description: str,
                 user_id: Optional[str] = None, agent_id: Optional[str] = None,
                 ip_address: Optional[str] = None, metadata: Optional[Dict] = None) -> str:
        """
        Log security event
        
        Args:
            event_type: Type of event
            severity: Event severity
            description: Event description
            user_id: User involved
            agent_id: Agent involved
            ip_address: IP address
            metadata: Additional metadata
            
        Returns:
            Event ID
        """
        self.event_counter += 1
        event_id = f"event_{self.event_counter}"
        
        event = SecurityEvent(
            event_id=event_id,
            event_type=event_type,
            severity=severity,
            description=description,
            user_id=user_id,
            agent_id=agent_id,
            ip_address=ip_address,
            metadata=metadata or {}
        )
        
        self.events.append(event)
        
        # Check for anomalies
        self._check_anomalies(event)
        
        if severity == "critical":
            logger.critical(f"Security event: {description}")
        elif severity == "warning":
            logger.warning(f"Security event: {description}")
        else:
            logger.info(f"Security event: {description}")
        
        return event_id
    
    def log_failed_authentication(self, user_id: str, ip_address: Optional[str] = None):
        """Log failed authentication attempt"""
        self.failed_auth_attempts[user_id].append(datetime.now())
        
        self.log_event(
            event_type="failed_authentication",
            severity="warning",
            description=f"Failed authentication for user {user_id}",
            user_id=user_id,
            ip_address=ip_address
        )
    
    def log_pii_access(self, user_id: str, document_id: str, pii_types: List[str]):
        """Log PII access"""
        self.pii_access_attempts[user_id].append(datetime.now())
        
        self.log_event(
            event_type="pii_access",
            severity="info",
            description=f"User {user_id} accessed PII in document {document_id}",
            user_id=user_id,
            metadata={
                'document_id': document_id,
                'pii_types': pii_types
            }
        )
    
    def log_unauthorized_access(self, user_id: str, resource_id: str, 
                               required_permissions: List[str]):
        """Log unauthorized access attempt"""
        self.log_event(
            event_type="unauthorized_access",
            severity="warning",
            description=f"User {user_id} attempted unauthorized access to {resource_id}",
            user_id=user_id,
            metadata={
                'resource_id': resource_id,
                'required_permissions': required_permissions
            }
        )
    
    def log_hallucination_detected(self, response_id: str, confidence: float, reason: str):
        """Log hallucination detection"""
        self.log_event(
            event_type="hallucination_detected",
            severity="warning",
            description=f"Potential hallucination detected in response {response_id}",
            metadata={
                'response_id': response_id,
                'confidence': confidence,
                'reason': reason
            }
        )
    
    def _check_anomalies(self, event: SecurityEvent):
        """Check for security anomalies"""
        # Check for brute force attacks
        if event.event_type == "failed_authentication" and event.user_id:
            recent_failures = [
                ts for ts in self.failed_auth_attempts[event.user_id]
                if ts > datetime.now() - timedelta(minutes=15)
            ]
            
            if len(recent_failures) >= 5:
                self._create_alert(
                    alert_type="brute_force_attack",
                    severity="critical",
                    message=f"Possible brute force attack on user {event.user_id}",
                    events=[event]
                )
        
        # Check for excessive PII access
        if event.event_type == "pii_access" and event.user_id:
            recent_access = [
                ts for ts in self.pii_access_attempts[event.user_id]
                if ts > datetime.now() - timedelta(hours=1)
            ]
            
            if len(recent_access) >= 20:
                self._create_alert(
                    alert_type="excessive_pii_access",
                    severity="warning",
                    message=f"User {event.user_id} accessing PII at unusual rate",
                    events=[event]
                )
        
        # Check for repeated unauthorized access
        if event.event_type == "unauthorized_access" and event.user_id:
            recent_unauthorized = [
                e for e in self.events[-50:]  # Check last 50 events
                if (e.event_type == "unauthorized_access" and 
                    e.user_id == event.user_id and
                    e.timestamp > datetime.now() - timedelta(minutes=30))
            ]
            
            if len(recent_unauthorized) >= 5:
                self._create_alert(
                    alert_type="privilege_escalation_attempt",
                    severity="critical",
                    message=f"Possible privilege escalation attempt by user {event.user_id}",
                    events=recent_unauthorized
                )
    
    def _create_alert(self, alert_type: str, severity: str, message: str,
                     events: List[SecurityEvent]):
        """Create security alert"""
        self.alert_counter += 1
        alert_id = f"alert_{self.alert_counter}"
        
        alert = SecurityAlert(
            alert_id=alert_id,
            alert_type=alert_type,
            severity=severity,
            message=message,
            events=events,
            triggered_at=datetime.now()
        )
        
        self.alerts.append(alert)
        
        logger.critical(f"SECURITY ALERT [{severity.upper()}]: {message}")
    
    def get_recent_events(self, minutes: int = 60, 
                         severity: Optional[str] = None) -> List[SecurityEvent]:
        """Get recent security events"""
        cutoff = datetime.now() - timedelta(minutes=minutes)
        
        recent = [e for e in self.events if e.timestamp > cutoff]
        
        if severity:
            recent = [e for e in recent if e.severity == severity]
        
        return recent
    
    def get_active_alerts(self) -> List[SecurityAlert]:
        """Get unacknowledged security alerts"""
        return [a for a in self.alerts if not a.acknowledged]
    
    def acknowledge_alert(self, alert_id: str) -> Dict[str, Any]:
        """Acknowledge security alert"""
        for alert in self.alerts:
            if alert.alert_id == alert_id:
                alert.acknowledged = True
                logger.info(f"Alert acknowledged: {alert_id}")
                return {'success': True}
        
        return {'success': False, 'error': 'Alert not found'}
    
    def get_security_summary(self) -> Dict[str, Any]:
        """Get security summary statistics"""
        recent_events = self.get_recent_events(minutes=60)
        
        event_counts = defaultdict(int)
        for event in recent_events:
            event_counts[event.event_type] += 1
        
        severity_counts = defaultdict(int)
        for event in recent_events:
            severity_counts[event.severity] += 1
        
        return {
            'recent_events': len(recent_events),
            'active_alerts': len(self.get_active_alerts()),
            'event_types': dict(event_counts),
            'severity_distribution': dict(severity_counts),
            'total_events': len(self.events),
            'total_alerts': len(self.alerts)
        }