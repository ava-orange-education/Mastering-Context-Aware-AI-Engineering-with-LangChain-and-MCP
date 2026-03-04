"""
Detect anomalies in security events and usage patterns.
"""

from typing import Dict, Any, List
from collections import defaultdict
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AnomalyDetector:
    """Detect anomalous patterns in system usage"""
    
    def __init__(self):
        """Initialize anomaly detector"""
        self.baseline_metrics: Dict[str, Any] = {}
        self.alert_thresholds = {
            'failed_auth_rate': 5,  # 5 failures per 15 min
            'query_rate': 100,  # 100 queries per hour
            'pii_access_rate': 20,  # 20 PII accesses per hour
            'error_rate': 10  # 10 errors per hour
        }
    
    def detect_failed_auth_spike(self, events: List[Any]) -> Dict[str, Any]:
        """
        Detect spike in failed authentication attempts
        
        Args:
            events: Recent security events
            
        Returns:
            Detection result
        """
        # Count failed auth in last 15 minutes
        cutoff = datetime.now() - timedelta(minutes=15)
        failed_auth = [
            e for e in events
            if e.event_type == 'failed_authentication' and e.timestamp > cutoff
        ]
        
        # Group by user
        by_user = defaultdict(int)
        for event in failed_auth:
            if event.user_id:
                by_user[event.user_id] += 1
        
        # Check for anomalies
        anomalies = []
        for user_id, count in by_user.items():
            if count >= self.alert_thresholds['failed_auth_rate']:
                anomalies.append({
                    'user_id': user_id,
                    'failed_attempts': count,
                    'timeframe': '15 minutes',
                    'severity': 'high' if count >= 10 else 'medium'
                })
        
        return {
            'anomaly_detected': len(anomalies) > 0,
            'anomalies': anomalies,
            'total_failed_auth': len(failed_auth)
        }
    
    def detect_excessive_queries(self, events: List[Any]) -> Dict[str, Any]:
        """Detect unusually high query rates"""
        cutoff = datetime.now() - timedelta(hours=1)
        queries = [
            e for e in events
            if e.event_type == 'query_executed' and e.timestamp > cutoff
        ]
        
        # Group by user
        by_user = defaultdict(int)
        for event in queries:
            if event.user_id:
                by_user[event.user_id] += 1
        
        # Check for anomalies
        anomalies = []
        for user_id, count in by_user.items():
            if count >= self.alert_thresholds['query_rate']:
                anomalies.append({
                    'user_id': user_id,
                    'query_count': count,
                    'timeframe': '1 hour',
                    'severity': 'medium'
                })
        
        return {
            'anomaly_detected': len(anomalies) > 0,
            'anomalies': anomalies
        }
    
    def detect_pii_access_pattern(self, events: List[Any]) -> Dict[str, Any]:
        """Detect unusual PII access patterns"""
        cutoff = datetime.now() - timedelta(hours=1)
        pii_access = [
            e for e in events
            if e.event_type == 'pii_access' and e.timestamp > cutoff
        ]
        
        by_user = defaultdict(int)
        for event in pii_access:
            if event.user_id:
                by_user[event.user_id] += 1
        
        anomalies = []
        for user_id, count in by_user.items():
            if count >= self.alert_thresholds['pii_access_rate']:
                anomalies.append({
                    'user_id': user_id,
                    'pii_accesses': count,
                    'timeframe': '1 hour',
                    'severity': 'high'
                })
        
        return {
            'anomaly_detected': len(anomalies) > 0,
            'anomalies': anomalies,
            'potential_data_exfiltration': len(anomalies) > 0
        }
    
    def run_all_detections(self, events: List[Any]) -> Dict[str, Any]:
        """Run all anomaly detections"""
        failed_auth = self.detect_failed_auth_spike(events)
        excessive_queries = self.detect_excessive_queries(events)
        pii_access = self.detect_pii_access_pattern(events)
        
        all_anomalies = (
            failed_auth['anomalies'] +
            excessive_queries['anomalies'] +
            pii_access['anomalies']
        )
        
        return {
            'total_anomalies': len(all_anomalies),
            'failed_auth': failed_auth,
            'excessive_queries': excessive_queries,
            'pii_access': pii_access,
            'requires_attention': len(all_anomalies) > 0
        }