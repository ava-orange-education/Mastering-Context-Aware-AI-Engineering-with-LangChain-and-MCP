"""
MTTR Metrics

Evaluates Mean Time To Resolve and related metrics
"""

from typing import Dict, Any, List, Optional
import logging
from datetime import datetime, timedelta
from collections import defaultdict
import statistics

logger = logging.getLogger(__name__)


class MTTRMetrics:
    """
    Calculates MTTR and related incident response metrics
    """
    
    def __init__(self):
        # Incident timing data
        self.incidents: List[Dict[str, Any]] = []
    
    def record_incident(
        self,
        incident_id: str,
        detected_at: datetime,
        acknowledged_at: Optional[datetime] = None,
        resolved_at: Optional[datetime] = None,
        severity: str = "medium",
        incident_type: Optional[str] = None
    ) -> None:
        """
        Record incident timing data
        
        Args:
            incident_id: Unique incident identifier
            detected_at: When incident was detected
            acknowledged_at: When incident was acknowledged
            resolved_at: When incident was resolved
            severity: Incident severity
            incident_type: Type of incident
        """
        
        incident = {
            "id": incident_id,
            "detected_at": detected_at,
            "acknowledged_at": acknowledged_at,
            "resolved_at": resolved_at,
            "severity": severity,
            "type": incident_type
        }
        
        self.incidents.append(incident)
        
        logger.info(f"Recorded incident: {incident_id}")
    
    def calculate_mttr(
        self,
        severity: Optional[str] = None,
        incident_type: Optional[str] = None,
        timeframe: Optional[timedelta] = None
    ) -> Optional[float]:
        """
        Calculate Mean Time To Resolve
        
        Args:
            severity: Filter by severity
            incident_type: Filter by type
            timeframe: Only include incidents within timeframe
        
        Returns:
            MTTR in seconds
        """
        
        # Filter incidents
        filtered = self._filter_incidents(
            severity=severity,
            incident_type=incident_type,
            timeframe=timeframe
        )
        
        # Calculate resolution times
        resolution_times = []
        
        for incident in filtered:
            if incident["resolved_at"] and incident["detected_at"]:
                resolution_time = (
                    incident["resolved_at"] - incident["detected_at"]
                ).total_seconds()
                resolution_times.append(resolution_time)
        
        if not resolution_times:
            return None
        
        mttr = statistics.mean(resolution_times)
        
        logger.info(
            f"MTTR: {mttr/60:.2f} minutes "
            f"(based on {len(resolution_times)} incidents)"
        )
        
        return mttr
    
    def calculate_mtta(
        self,
        severity: Optional[str] = None,
        incident_type: Optional[str] = None,
        timeframe: Optional[timedelta] = None
    ) -> Optional[float]:
        """
        Calculate Mean Time To Acknowledge
        
        Args:
            severity: Filter by severity
            incident_type: Filter by type
            timeframe: Only include incidents within timeframe
        
        Returns:
            MTTA in seconds
        """
        
        filtered = self._filter_incidents(
            severity=severity,
            incident_type=incident_type,
            timeframe=timeframe
        )
        
        acknowledgment_times = []
        
        for incident in filtered:
            if incident["acknowledged_at"] and incident["detected_at"]:
                ack_time = (
                    incident["acknowledged_at"] - incident["detected_at"]
                ).total_seconds()
                acknowledgment_times.append(ack_time)
        
        if not acknowledgment_times:
            return None
        
        mtta = statistics.mean(acknowledgment_times)
        
        logger.info(
            f"MTTA: {mtta/60:.2f} minutes "
            f"(based on {len(acknowledgment_times)} incidents)"
        )
        
        return mtta
    
    def calculate_mttr_by_severity(self) -> Dict[str, float]:
        """Calculate MTTR broken down by severity"""
        
        mttr_by_severity = {}
        
        severities = set(inc["severity"] for inc in self.incidents)
        
        for severity in severities:
            mttr = self.calculate_mttr(severity=severity)
            if mttr is not None:
                mttr_by_severity[severity] = mttr
        
        return mttr_by_severity
    
    def calculate_mttr_by_type(self) -> Dict[str, float]:
        """Calculate MTTR broken down by incident type"""
        
        mttr_by_type = {}
        
        types = set(
            inc["type"] for inc in self.incidents
            if inc["type"] is not None
        )
        
        for incident_type in types:
            mttr = self.calculate_mttr(incident_type=incident_type)
            if mttr is not None:
                mttr_by_type[incident_type] = mttr
        
        return mttr_by_type
    
    def get_percentile_resolution_time(
        self,
        percentile: float = 0.95,
        severity: Optional[str] = None
    ) -> Optional[float]:
        """
        Get percentile resolution time (e.g., P95)
        
        Args:
            percentile: Percentile (0.0 to 1.0)
            severity: Optional severity filter
        
        Returns:
            Resolution time at percentile
        """
        
        filtered = self._filter_incidents(severity=severity)
        
        resolution_times = []
        
        for incident in filtered:
            if incident["resolved_at"] and incident["detected_at"]:
                resolution_time = (
                    incident["resolved_at"] - incident["detected_at"]
                ).total_seconds()
                resolution_times.append(resolution_time)
        
        if not resolution_times:
            return None
        
        sorted_times = sorted(resolution_times)
        index = int(len(sorted_times) * percentile)
        
        if index >= len(sorted_times):
            index = len(sorted_times) - 1
        
        return sorted_times[index]
    
    def get_incident_volume(
        self,
        timeframe: timedelta = timedelta(days=7)
    ) -> Dict[str, Any]:
        """
        Get incident volume statistics
        
        Args:
            timeframe: Time window
        
        Returns:
            Volume statistics
        """
        
        filtered = self._filter_incidents(timeframe=timeframe)
        
        # Count by severity
        by_severity = defaultdict(int)
        for incident in filtered:
            by_severity[incident["severity"]] += 1
        
        # Count by type
        by_type = defaultdict(int)
        for incident in filtered:
            if incident["type"]:
                by_type[incident["type"]] += 1
        
        return {
            "total_incidents": len(filtered),
            "timeframe_days": timeframe.days,
            "by_severity": dict(by_severity),
            "by_type": dict(by_type),
            "incidents_per_day": len(filtered) / max(timeframe.days, 1)
        }
    
    def calculate_sla_compliance(
        self,
        sla_targets: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Calculate SLA compliance
        
        Args:
            sla_targets: SLA targets by severity (in seconds)
                Example: {"critical": 300, "high": 1800}
        
        Returns:
            SLA compliance metrics
        """
        
        compliance = {}
        
        for severity, target_time in sla_targets.items():
            filtered = self._filter_incidents(severity=severity)
            
            if not filtered:
                continue
            
            within_sla = 0
            total = 0
            
            for incident in filtered:
                if incident["resolved_at"] and incident["detected_at"]:
                    resolution_time = (
                        incident["resolved_at"] - incident["detected_at"]
                    ).total_seconds()
                    
                    total += 1
                    if resolution_time <= target_time:
                        within_sla += 1
            
            if total > 0:
                compliance[severity] = {
                    "target_seconds": target_time,
                    "total_incidents": total,
                    "within_sla": within_sla,
                    "compliance_rate": within_sla / total
                }
        
        return compliance
    
    def _filter_incidents(
        self,
        severity: Optional[str] = None,
        incident_type: Optional[str] = None,
        timeframe: Optional[timedelta] = None
    ) -> List[Dict[str, Any]]:
        """Filter incidents by criteria"""
        
        filtered = self.incidents.copy()
        
        if severity:
            filtered = [
                inc for inc in filtered
                if inc["severity"] == severity
            ]
        
        if incident_type:
            filtered = [
                inc for inc in filtered
                if inc["type"] == incident_type
            ]
        
        if timeframe:
            cutoff = datetime.utcnow() - timeframe
            filtered = [
                inc for inc in filtered
                if inc["detected_at"] >= cutoff
            ]
        
        return filtered
    
    def get_summary(self) -> Dict[str, Any]:
        """Get comprehensive MTTR summary"""
        
        return {
            "total_incidents": len(self.incidents),
            "mttr_overall": self.calculate_mttr(),
            "mtta_overall": self.calculate_mtta(),
            "mttr_by_severity": self.calculate_mttr_by_severity(),
            "mttr_by_type": self.calculate_mttr_by_type(),
            "p95_resolution_time": self.get_percentile_resolution_time(0.95),
            "p99_resolution_time": self.get_percentile_resolution_time(0.99)
        }