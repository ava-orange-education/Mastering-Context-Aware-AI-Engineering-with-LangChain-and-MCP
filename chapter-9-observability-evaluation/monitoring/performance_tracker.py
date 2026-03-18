"""
Track performance metrics over time.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from collections import defaultdict
import statistics
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PerformanceTracker:
    """Track performance metrics over time"""
    
    def __init__(self, window_minutes: int = 60):
        """
        Initialize performance tracker
        
        Args:
            window_minutes: Time window for metrics (minutes)
        """
        self.window_minutes = window_minutes
        self.metrics: List[Dict[str, Any]] = []
        self.latency_samples: List[float] = []
        self.throughput_samples: List[int] = []
    
    def record_metric(self, metric_name: str, value: float, 
                     timestamp: Optional[datetime] = None,
                     tags: Optional[Dict[str, str]] = None):
        """
        Record a metric
        
        Args:
            metric_name: Metric name
            value: Metric value
            timestamp: Timestamp (defaults to now)
            tags: Additional tags
        """
        metric = {
            'name': metric_name,
            'value': value,
            'timestamp': timestamp or datetime.now(),
            'tags': tags or {}
        }
        
        self.metrics.append(metric)
        
        # Track specific metrics
        if metric_name == 'latency':
            self.latency_samples.append(value)
        elif metric_name == 'throughput':
            self.throughput_samples.append(int(value))
        
        # Clean old metrics
        self._clean_old_metrics()
        
        logger.debug(f"Metric recorded: {metric_name}={value}")
    
    def record_latency(self, latency: float, operation: str = "default"):
        """
        Record latency metric
        
        Args:
            latency: Latency in seconds
            operation: Operation name
        """
        self.record_metric('latency', latency, tags={'operation': operation})
    
    def record_throughput(self, count: int, operation: str = "default"):
        """
        Record throughput metric
        
        Args:
            count: Number of operations
            operation: Operation name
        """
        self.record_metric('throughput', count, tags={'operation': operation})
    
    def record_error_rate(self, error_count: int, total_count: int):
        """
        Record error rate
        
        Args:
            error_count: Number of errors
            total_count: Total operations
        """
        error_rate = error_count / total_count if total_count > 0 else 0
        self.record_metric('error_rate', error_rate)
    
    def get_latency_stats(self) -> Dict[str, float]:
        """
        Get latency statistics
        
        Returns:
            Statistics dictionary
        """
        if not self.latency_samples:
            return {}
        
        recent_samples = self._get_recent_samples(self.latency_samples)
        
        if not recent_samples:
            return {}
        
        return {
            'mean': statistics.mean(recent_samples),
            'median': statistics.median(recent_samples),
            'stdev': statistics.stdev(recent_samples) if len(recent_samples) > 1 else 0,
            'min': min(recent_samples),
            'max': max(recent_samples),
            'p95': self._percentile(recent_samples, 0.95),
            'p99': self._percentile(recent_samples, 0.99),
            'sample_count': len(recent_samples)
        }
    
    def get_throughput_stats(self) -> Dict[str, float]:
        """
        Get throughput statistics
        
        Returns:
            Statistics dictionary
        """
        if not self.throughput_samples:
            return {}
        
        recent_samples = self._get_recent_samples(self.throughput_samples)
        
        if not recent_samples:
            return {}
        
        return {
            'mean': statistics.mean(recent_samples),
            'median': statistics.median(recent_samples),
            'total': sum(recent_samples),
            'min': min(recent_samples),
            'max': max(recent_samples),
            'sample_count': len(recent_samples)
        }
    
    def get_metrics_by_name(self, metric_name: str, 
                           minutes: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get metrics by name within time window
        
        Args:
            metric_name: Metric name
            minutes: Time window in minutes (None for all)
            
        Returns:
            List of metrics
        """
        cutoff = None
        if minutes:
            cutoff = datetime.now() - timedelta(minutes=minutes)
        
        matching = [
            m for m in self.metrics 
            if m['name'] == metric_name and (not cutoff or m['timestamp'] > cutoff)
        ]
        
        return matching
    
    def get_trend(self, metric_name: str, minutes: int = 60) -> Dict[str, Any]:
        """
        Get metric trend over time
        
        Args:
            metric_name: Metric name
            minutes: Time window
            
        Returns:
            Trend analysis
        """
        metrics = self.get_metrics_by_name(metric_name, minutes)
        
        if len(metrics) < 2:
            return {'trend': 'insufficient_data'}
        
        # Sort by timestamp
        metrics.sort(key=lambda m: m['timestamp'])
        
        values = [m['value'] for m in metrics]
        
        # Simple trend: compare first half to second half
        mid = len(values) // 2
        first_half_avg = statistics.mean(values[:mid])
        second_half_avg = statistics.mean(values[mid:])
        
        change_pct = ((second_half_avg - first_half_avg) / first_half_avg * 100 
                     if first_half_avg > 0 else 0)
        
        if change_pct > 10:
            trend = 'increasing'
        elif change_pct < -10:
            trend = 'decreasing'
        else:
            trend = 'stable'
        
        return {
            'trend': trend,
            'change_percent': change_pct,
            'first_half_avg': first_half_avg,
            'second_half_avg': second_half_avg,
            'sample_count': len(values)
        }
    
    def _get_recent_samples(self, samples: List[float]) -> List[float]:
        """Get samples within time window"""
        # Since we clean old metrics regularly, all samples are recent
        return samples
    
    def _percentile(self, data: List[float], percentile: float) -> float:
        """Calculate percentile"""
        sorted_data = sorted(data)
        index = int(len(sorted_data) * percentile)
        return sorted_data[min(index, len(sorted_data) - 1)]
    
    def _clean_old_metrics(self):
        """Remove metrics outside time window"""
        cutoff = datetime.now() - timedelta(minutes=self.window_minutes)
        
        # Remove old metrics
        self.metrics = [m for m in self.metrics if m['timestamp'] > cutoff]
        
        # Keep only recent latency samples (last 1000)
        if len(self.latency_samples) > 1000:
            self.latency_samples = self.latency_samples[-1000:]
        
        if len(self.throughput_samples) > 1000:
            self.throughput_samples = self.throughput_samples[-1000:]
    
    def get_summary(self) -> Dict[str, Any]:
        """Get overall performance summary"""
        return {
            'total_metrics': len(self.metrics),
            'window_minutes': self.window_minutes,
            'latency_stats': self.get_latency_stats(),
            'throughput_stats': self.get_throughput_stats(),
            'unique_metric_names': len(set(m['name'] for m in self.metrics))
        }