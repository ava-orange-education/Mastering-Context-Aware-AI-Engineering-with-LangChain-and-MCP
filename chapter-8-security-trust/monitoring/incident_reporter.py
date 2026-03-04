"""
Security incident reporting and tracking.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SecurityIncident:
    """Security incident record"""
    incident_id: str
    incident_type: str
    severity: str
    description: str
    detected_at: datetime
    affected_users: List[str] = field(default_factory=list)
    affected_resources: List[str] = field(default_factory=list)
    status: str = "open"  # open, investigating, resolved, closed
    resolution: Optional[str] = None
    resolved_at: Optional[datetime] = None


class IncidentReporter:
    """Report and track security incidents"""
    
    def __init__(self):
        """Initialize incident reporter"""
        self.incidents: List[SecurityIncident] = []
        self.incident_counter = 0
    
    def report_incident(self, incident_type: str, severity: str,
                       description: str, affected_users: Optional[List[str]] = None,
                       affected_resources: Optional[List[str]] = None) -> str:
        """
        Report new security incident
        
        Args:
            incident_type: Type of incident
            severity: Severity level
            description: Incident description
            affected_users: Affected user IDs
            affected_resources: Affected resource IDs
            
        Returns:
            Incident ID
        """
        self.incident_counter += 1
        incident_id = f"INC-{datetime.now().strftime('%Y%m%d')}-{self.incident_counter:04d}"
        
        incident = SecurityIncident(
            incident_id=incident_id,
            incident_type=incident_type,
            severity=severity,
            description=description,
            detected_at=datetime.now(),
            affected_users=affected_users or [],
            affected_resources=affected_resources or []
        )
        
        self.incidents.append(incident)
        
        logger.critical(f"Security incident reported: {incident_id} - {description}")
        
        return incident_id
    
    def update_incident_status(self, incident_id: str, status: str,
                              resolution: Optional[str] = None):
        """
        Update incident status
        
        Args:
            incident_id: Incident to update
            status: New status
            resolution: Resolution details
        """
        for incident in self.incidents:
            if incident.incident_id == incident_id:
                incident.status = status
                
                if resolution:
                    incident.resolution = resolution
                
                if status in ['resolved', 'closed']:
                    incident.resolved_at = datetime.now()
                
                logger.info(f"Incident {incident_id} updated: {status}")
                return
        
        logger.warning(f"Incident {incident_id} not found")
    
    def get_open_incidents(self) -> List[SecurityIncident]:
        """Get all open incidents"""
        return [i for i in self.incidents if i.status == 'open']
    
    def get_incidents_by_severity(self, severity: str) -> List[SecurityIncident]:
        """Get incidents by severity"""
        return [i for i in self.incidents if i.severity == severity]
    
    def generate_incident_report(self, days: int = 30) -> Dict[str, Any]:
        """
        Generate incident summary report
        
        Args:
            days: Report period in days
            
        Returns:
            Incident summary
        """
        from datetime import timedelta
        
        cutoff = datetime.now() - timedelta(days=days)
        recent = [i for i in self.incidents if i.detected_at >= cutoff]
        
        # Count by type
        by_type = {}
        for incident in recent:
            by_type[incident.incident_type] = by_type.get(incident.incident_type, 0) + 1
        
        # Count by severity
        by_severity = {}
        for incident in recent:
            by_severity[incident.severity] = by_severity.get(incident.severity, 0) + 1
        
        # Count by status
        by_status = {}
        for incident in recent:
            by_status[incident.status] = by_status.get(incident.status, 0) + 1
        
        return {
            'period_days': days,
            'total_incidents': len(recent),
            'by_type': by_type,
            'by_severity': by_severity,
            'by_status': by_status,
            'open_incidents': len([i for i in recent if i.status == 'open']),
            'critical_incidents': len([i for i in recent if i.severity == 'critical'])
        }