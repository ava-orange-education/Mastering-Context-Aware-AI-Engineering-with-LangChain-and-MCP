"""
Track and analyze errors.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from collections import defaultdict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ErrorTracker:
    """Track and analyze errors"""
    
    def __init__(self):
        """Initialize error tracker"""
        self.errors: List[Dict[str, Any]] = []
        self.error_counter = 0
    
    def log_error(self, error_type: str, error_message: str,
                 component: str = "unknown", severity: str = "error",
                 context: Optional[Dict] = None):
        """
        Log an error
        
        Args:
            error_type: Error type/category
            error_message: Error message
            component: Component where error occurred
            severity: Error severity (info, warning, error, critical)
            context: Additional context
        """
        self.error_counter += 1
        
        error_record = {
            'error_id': f"err_{self.error_counter}",
            'error_type': error_type,
            'message': error_message,
            'component': component,
            'severity': severity,
            'timestamp': datetime.now(),
            'context': context or {}
        }
        
        self.errors.append(error_record)
        
        log_method = getattr(logger, severity, logger.error)
        log_method(f"Error logged: {error_type} in {component} - {error_message}")
    
    def get_error_count(self, minutes: Optional[int] = None,
                       error_type: Optional[str] = None,
                       severity: Optional[str] = None) -> int:
        """
        Get error count with filters
        
        Args:
            minutes: Time window in minutes
            error_type: Filter by error type
            severity: Filter by severity
            
        Returns:
            Error count
        """
        cutoff = datetime.now() - timedelta(minutes=minutes) if minutes else None
        
        filtered = self.errors
        
        if cutoff:
            filtered = [e for e in filtered if e['timestamp'] > cutoff]
        
        if error_type:
            filtered = [e for e in filtered if e['error_type'] == error_type]
        
        if severity:
            filtered = [e for e in filtered if e['severity'] == severity]
        
        return len(filtered)
    
    def get_error_rate(self, minutes: int = 60) -> Dict[str, Any]:
        """
        Calculate error rate over time window
        
        Args:
            minutes: Time window
            
        Returns:
            Error rate statistics
        """
        cutoff = datetime.now() - timedelta(minutes=minutes)
        recent_errors = [e for e in self.errors if e['timestamp'] > cutoff]
        
        if not recent_errors:
            return {
                'error_count': 0,
                'errors_per_minute': 0,
                'window_minutes': minutes
            }
        
        error_count = len(recent_errors)
        errors_per_minute = error_count / minutes
        
        return {
            'error_count': error_count,
            'errors_per_minute': errors_per_minute,
            'window_minutes': minutes
        }
    
    def get_errors_by_type(self, minutes: Optional[int] = None) -> Dict[str, int]:
        """
        Get error counts grouped by type
        
        Args:
            minutes: Time window in minutes
            
        Returns:
            Error counts by type
        """
        cutoff = datetime.now() - timedelta(minutes=minutes) if minutes else None
        
        errors_to_count = [e for e in self.errors if not cutoff or e['timestamp'] > cutoff]
        
        counts = defaultdict(int)
        for error in errors_to_count:
            counts[error['error_type']] += 1
        
        return dict(counts)
    
    def get_errors_by_component(self, minutes: Optional[int] = None) -> Dict[str, int]:
        """
        Get error counts grouped by component
        
        Args:
            minutes: Time window
            
        Returns:
            Error counts by component
        """
        cutoff = datetime.now() - timedelta(minutes=minutes) if minutes else None
        
        errors_to_count = [e for e in self.errors if not cutoff or e['timestamp'] > cutoff]
        
        counts = defaultdict(int)
        for error in errors_to_count:
            counts[error['component']] += 1
        
        return dict(counts)
    
    def get_top_errors(self, count: int = 10, minutes: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get most common errors
        
        Args:
            count: Number of top errors
            minutes: Time window
            
        Returns:
            List of top errors with counts
        """
        error_types = self.get_errors_by_type(minutes)
        
        sorted_errors = sorted(error_types.items(), key=lambda x: x[1], reverse=True)
        
        return [
            {'error_type': error_type, 'count': error_count}
            for error_type, error_count in sorted_errors[:count]
        ]
    
    def get_recent_errors(self, count: int = 10, 
                         severity: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get most recent errors
        
        Args:
            count: Number of errors
            severity: Filter by severity
            
        Returns:
            List of recent errors
        """
        errors = self.errors
        
        if severity:
            errors = [e for e in errors if e['severity'] == severity]
        
        # Sort by timestamp descending
        errors = sorted(errors, key=lambda e: e['timestamp'], reverse=True)
        
        return errors[:count]
    
    def detect_error_spike(self, threshold: float = 2.0, 
                          window_minutes: int = 10) -> Dict[str, Any]:
        """
        Detect if error rate is spiking
        
        Args:
            threshold: Spike threshold multiplier
            window_minutes: Time window for comparison
            
        Returns:
            Spike detection result
        """
        # Compare recent window to previous window
        now = datetime.now()
        recent_window = now - timedelta(minutes=window_minutes)
        previous_window = recent_window - timedelta(minutes=window_minutes)
        
        recent_errors = [
            e for e in self.errors 
            if recent_window <= e['timestamp'] <= now
        ]
        
        previous_errors = [
            e for e in self.errors 
            if previous_window <= e['timestamp'] < recent_window
        ]
        
        recent_count = len(recent_errors)
        previous_count = len(previous_errors)
        
        if previous_count == 0:
            spike_detected = recent_count > 5  # Absolute threshold
            multiplier = float('inf') if recent_count > 0 else 0
        else:
            multiplier = recent_count / previous_count
            spike_detected = multiplier >= threshold
        
        return {
            'spike_detected': spike_detected,
            'recent_count': recent_count,
            'previous_count': previous_count,
            'multiplier': multiplier,
            'threshold': threshold
        }
    
    def get_summary(self) -> Dict[str, Any]:
        """Get error tracking summary"""
        return {
            'total_errors': len(self.errors),
            'errors_last_hour': self.get_error_count(minutes=60),
            'errors_by_type': self.get_errors_by_type(minutes=60),
            'errors_by_severity': self._get_errors_by_severity(),
            'top_errors': self.get_top_errors(5, minutes=60)
        }
    
    def _get_errors_by_severity(self) -> Dict[str, int]:
        """Get error counts by severity"""
        counts = defaultdict(int)
        for error in self.errors:
            counts[error['severity']] += 1
        return dict(counts)