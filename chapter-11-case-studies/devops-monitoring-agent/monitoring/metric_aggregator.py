"""
Metric Aggregator

Aggregates and processes metrics from multiple sources
"""

from typing import Dict, Any, List, Optional
import logging
from datetime import datetime, timedelta
from collections import defaultdict
import statistics

logger = logging.getLogger(__name__)


class MetricAggregator:
    """
    Aggregates metrics from multiple sources
    """
    
    def __init__(self):
        # Metric storage
        self.metrics: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        
        # Retention period
        self.retention_minutes = 60
        
        # Aggregation windows
        self.windows = {
            "1m": 60,
            "5m": 300,
            "15m": 900,
            "1h": 3600
        }
    
    def add_metric(
        self,
        metric_name: str,
        value: float,
        timestamp: Optional[datetime] = None,
        labels: Optional[Dict[str, str]] = None
    ) -> None:
        """
        Add a metric data point
        
        Args:
            metric_name: Metric name
            value: Metric value
            timestamp: Timestamp (defaults to now)
            labels: Optional labels/tags
        """
        
        if timestamp is None:
            timestamp = datetime.utcnow()
        
        data_point = {
            "value": value,
            "timestamp": timestamp,
            "labels": labels or {}
        }
        
        self.metrics[metric_name].append(data_point)
        
        # Clean old data
        self._cleanup_old_metrics(metric_name)
    
    def add_metrics_batch(
        self,
        metrics: Dict[str, float],
        timestamp: Optional[datetime] = None
    ) -> None:
        """Add multiple metrics at once"""
        
        for metric_name, value in metrics.items():
            self.add_metric(metric_name, value, timestamp)
    
    def get_metric(
        self,
        metric_name: str,
        window: str = "5m",
        aggregation: str = "avg"
    ) -> Optional[float]:
        """
        Get aggregated metric value
        
        Args:
            metric_name: Metric name
            window: Time window (1m, 5m, 15m, 1h)
            aggregation: Aggregation function (avg, sum, min, max, count)
        
        Returns:
            Aggregated value
        """
        
        if metric_name not in self.metrics:
            return None
        
        # Get data points in window
        window_seconds = self.windows.get(window, 300)
        cutoff = datetime.utcnow() - timedelta(seconds=window_seconds)
        
        data_points = [
            dp for dp in self.metrics[metric_name]
            if dp["timestamp"] >= cutoff
        ]
        
        if not data_points:
            return None
        
        values = [dp["value"] for dp in data_points]
        
        # Aggregate
        if aggregation == "avg":
            return statistics.mean(values)
        elif aggregation == "sum":
            return sum(values)
        elif aggregation == "min":
            return min(values)
        elif aggregation == "max":
            return max(values)
        elif aggregation == "count":
            return len(values)
        elif aggregation == "p50":
            return statistics.median(values)
        elif aggregation == "p95":
            return self._percentile(values, 0.95)
        elif aggregation == "p99":
            return self._percentile(values, 0.99)
        else:
            return statistics.mean(values)
    
    def get_metrics(
        self,
        metric_names: List[str],
        window: str = "5m",
        aggregation: str = "avg"
    ) -> Dict[str, Optional[float]]:
        """Get multiple metrics"""
        
        return {
            name: self.get_metric(name, window, aggregation)
            for name in metric_names
        }
    
    def get_rate(
        self,
        metric_name: str,
        window: str = "5m"
    ) -> Optional[float]:
        """
        Calculate rate of change for a counter metric
        
        Args:
            metric_name: Counter metric name
            window: Time window
        
        Returns:
            Rate per second
        """
        
        if metric_name not in self.metrics:
            return None
        
        window_seconds = self.windows.get(window, 300)
        cutoff = datetime.utcnow() - timedelta(seconds=window_seconds)
        
        data_points = [
            dp for dp in self.metrics[metric_name]
            if dp["timestamp"] >= cutoff
        ]
        
        if len(data_points) < 2:
            return None
        
        # Sort by timestamp
        data_points.sort(key=lambda x: x["timestamp"])
        
        # Calculate rate
        first = data_points[0]
        last = data_points[-1]
        
        value_diff = last["value"] - first["value"]
        time_diff = (last["timestamp"] - first["timestamp"]).total_seconds()
        
        if time_diff == 0:
            return None
        
        return value_diff / time_diff
    
    def calculate_delta(
        self,
        metric_name: str,
        window1: str = "5m",
        window2: str = "15m"
    ) -> Optional[float]:
        """
        Calculate difference between two time windows
        
        Args:
            metric_name: Metric name
            window1: Recent window
            window2: Older window
        
        Returns:
            Delta (window1 - window2)
        """
        
        recent = self.get_metric(metric_name, window1)
        older = self.get_metric(metric_name, window2)
        
        if recent is None or older is None:
            return None
        
        return recent - older
    
    def get_trend(
        self,
        metric_name: str,
        window: str = "15m"
    ) -> Optional[str]:
        """
        Determine metric trend
        
        Args:
            metric_name: Metric name
            window: Time window
        
        Returns:
            Trend: "increasing", "decreasing", "stable"
        """
        
        if metric_name not in self.metrics:
            return None
        
        window_seconds = self.windows.get(window, 900)
        cutoff = datetime.utcnow() - timedelta(seconds=window_seconds)
        
        data_points = [
            dp for dp in self.metrics[metric_name]
            if dp["timestamp"] >= cutoff
        ]
        
        if len(data_points) < 5:
            return None
        
        # Sort by timestamp
        data_points.sort(key=lambda x: x["timestamp"])
        values = [dp["value"] for dp in data_points]
        
        # Simple linear regression
        n = len(values)
        x = list(range(n))
        
        x_mean = sum(x) / n
        y_mean = sum(values) / n
        
        numerator = sum((x[i] - x_mean) * (values[i] - y_mean) for i in range(n))
        denominator = sum((x[i] - x_mean) ** 2 for i in range(n))
        
        if denominator == 0:
            return "stable"
        
        slope = numerator / denominator
        
        # Determine trend based on slope
        if slope > 0.05:
            return "increasing"
        elif slope < -0.05:
            return "decreasing"
        else:
            return "stable"
    
    def _percentile(self, values: List[float], percentile: float) -> float:
        """Calculate percentile"""
        
        sorted_values = sorted(values)
        index = int(len(sorted_values) * percentile)
        
        if index >= len(sorted_values):
            index = len(sorted_values) - 1
        
        return sorted_values[index]
    
    def _cleanup_old_metrics(self, metric_name: str) -> None:
        """Remove old metric data points"""
        
        cutoff = datetime.utcnow() - timedelta(minutes=self.retention_minutes)
        
        self.metrics[metric_name] = [
            dp for dp in self.metrics[metric_name]
            if dp["timestamp"] >= cutoff
        ]
    
    def get_statistics(
        self,
        metric_name: str,
        window: str = "15m"
    ) -> Optional[Dict[str, float]]:
        """
        Get statistical summary of metric
        
        Args:
            metric_name: Metric name
            window: Time window
        
        Returns:
            Statistics dict
        """
        
        if metric_name not in self.metrics:
            return None
        
        window_seconds = self.windows.get(window, 900)
        cutoff = datetime.utcnow() - timedelta(seconds=window_seconds)
        
        data_points = [
            dp for dp in self.metrics[metric_name]
            if dp["timestamp"] >= cutoff
        ]
        
        if not data_points:
            return None
        
        values = [dp["value"] for dp in data_points]
        
        return {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "stdev": statistics.stdev(values) if len(values) > 1 else 0,
            "p95": self._percentile(values, 0.95),
            "p99": self._percentile(values, 0.99)
        }
    
    def get_all_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of all metrics"""
        
        summary = {
            "total_metrics": len(self.metrics),
            "metrics": {}
        }
        
        for metric_name in self.metrics:
            stats = self.get_statistics(metric_name)
            if stats:
                summary["metrics"][metric_name] = {
                    "current": self.get_metric(metric_name),
                    "trend": self.get_trend(metric_name),
                    "data_points": stats["count"]
                }
        
        return summary