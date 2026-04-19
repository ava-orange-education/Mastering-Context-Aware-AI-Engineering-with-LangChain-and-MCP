"""
Anomaly Detector

Detects anomalies in metrics using statistical methods
"""

from typing import Dict, Any, List, Optional, Tuple
import logging
from datetime import datetime
from collections import deque
import statistics

logger = logging.getLogger(__name__)


class AnomalyDetector:
    """
    Detects anomalies in time-series metrics
    """
    
    def __init__(self):
        # Historical data for baseline
        self.history: Dict[str, deque] = {}
        self.max_history = 1000
        
        # Detection thresholds
        self.z_score_threshold = 3.0
        self.percentage_change_threshold = 0.5  # 50%
    
    def add_datapoint(
        self,
        metric_name: str,
        value: float,
        timestamp: Optional[datetime] = None
    ) -> None:
        """
        Add a datapoint to history
        
        Args:
            metric_name: Metric name
            value: Metric value
            timestamp: Timestamp (defaults to now)
        """
        
        if metric_name not in self.history:
            self.history[metric_name] = deque(maxlen=self.max_history)
        
        self.history[metric_name].append({
            "value": value,
            "timestamp": timestamp or datetime.utcnow()
        })
    
    def detect_anomalies(
        self,
        metrics: Dict[str, float]
    ) -> List[Dict[str, Any]]:
        """
        Detect anomalies in current metrics
        
        Args:
            metrics: Current metric values
        
        Returns:
            List of detected anomalies
        """
        
        anomalies = []
        
        for metric_name, current_value in metrics.items():
            # Add to history
            self.add_datapoint(metric_name, current_value)
            
            # Check for anomalies
            anomaly = self._check_metric_anomaly(metric_name, current_value)
            
            if anomaly:
                anomalies.append(anomaly)
        
        return anomalies
    
    def _check_metric_anomaly(
        self,
        metric_name: str,
        current_value: float
    ) -> Optional[Dict[str, Any]]:
        """Check if a metric value is anomalous"""
        
        if metric_name not in self.history:
            return None
        
        history = self.history[metric_name]
        
        if len(history) < 10:
            # Not enough history
            return None
        
        # Extract historical values
        historical_values = [dp["value"] for dp in history][:-1]  # Exclude current
        
        # Method 1: Z-score (standard deviations from mean)
        z_score_anomaly = self._detect_zscore_anomaly(
            metric_name,
            current_value,
            historical_values
        )
        
        if z_score_anomaly:
            return z_score_anomaly
        
        # Method 2: Percentage change from recent average
        percentage_anomaly = self._detect_percentage_anomaly(
            metric_name,
            current_value,
            historical_values
        )
        
        if percentage_anomaly:
            return percentage_anomaly
        
        # Method 3: Trend-based detection
        trend_anomaly = self._detect_trend_anomaly(
            metric_name,
            current_value,
            historical_values
        )
        
        if trend_anomaly:
            return trend_anomaly
        
        return None
    
    def _detect_zscore_anomaly(
        self,
        metric_name: str,
        current_value: float,
        historical_values: List[float]
    ) -> Optional[Dict[str, Any]]:
        """Detect anomaly using z-score"""
        
        if len(historical_values) < 2:
            return None
        
        try:
            mean = statistics.mean(historical_values)
            stdev = statistics.stdev(historical_values)
            
            if stdev == 0:
                return None
            
            z_score = abs((current_value - mean) / stdev)
            
            if z_score > self.z_score_threshold:
                return {
                    "metric": metric_name,
                    "type": "statistical_anomaly",
                    "method": "z_score",
                    "current_value": current_value,
                    "expected_value": mean,
                    "z_score": round(z_score, 2),
                    "severity": "high" if z_score > 5 else "medium",
                    "description": f"{metric_name} is {z_score:.1f} standard deviations from mean"
                }
        
        except statistics.StatisticsError:
            pass
        
        return None
    
    def _detect_percentage_anomaly(
        self,
        metric_name: str,
        current_value: float,
        historical_values: List[float]
    ) -> Optional[Dict[str, Any]]:
        """Detect anomaly using percentage change"""
        
        # Use recent average (last 10 points)
        recent_values = historical_values[-10:]
        
        if not recent_values:
            return None
        
        recent_avg = statistics.mean(recent_values)
        
        if recent_avg == 0:
            return None
        
        percentage_change = abs((current_value - recent_avg) / recent_avg)
        
        if percentage_change > self.percentage_change_threshold:
            return {
                "metric": metric_name,
                "type": "percentage_anomaly",
                "method": "percentage_change",
                "current_value": current_value,
                "expected_value": recent_avg,
                "percentage_change": round(percentage_change * 100, 1),
                "severity": "high" if percentage_change > 1.0 else "medium",
                "description": f"{metric_name} changed {percentage_change*100:.0f}% from recent average"
            }
        
        return None
    
    def _detect_trend_anomaly(
        self,
        metric_name: str,
        current_value: float,
        historical_values: List[float]
    ) -> Optional[Dict[str, Any]]:
        """Detect anomaly based on trend"""
        
        if len(historical_values) < 20:
            return None
        
        # Calculate recent trend
        recent_values = historical_values[-20:]
        
        # Simple linear trend
        n = len(recent_values)
        x = list(range(n))
        y = recent_values
        
        x_mean = sum(x) / n
        y_mean = sum(y) / n
        
        numerator = sum((x[i] - x_mean) * (y[i] - y_mean) for i in range(n))
        denominator = sum((x[i] - x_mean) ** 2 for i in range(n))
        
        if denominator == 0:
            return None
        
        slope = numerator / denominator
        
        # Predict next value based on trend
        predicted = y_mean + slope * (n - x_mean)
        
        # Check if current value deviates significantly from prediction
        if predicted != 0:
            deviation = abs((current_value - predicted) / predicted)
            
            if deviation > 0.3:  # 30% deviation from trend
                return {
                    "metric": metric_name,
                    "type": "trend_anomaly",
                    "method": "trend_deviation",
                    "current_value": current_value,
                    "expected_value": predicted,
                    "deviation": round(deviation * 100, 1),
                    "trend_slope": round(slope, 4),
                    "severity": "medium",
                    "description": f"{metric_name} deviates {deviation*100:.0f}% from trend"
                }
        
        return None
    
    def get_baseline_stats(
        self,
        metric_name: str
    ) -> Optional[Dict[str, Any]]:
        """Get baseline statistics for a metric"""
        
        if metric_name not in self.history:
            return None
        
        values = [dp["value"] for dp in self.history[metric_name]]
        
        if len(values) < 2:
            return None
        
        return {
            "metric": metric_name,
            "count": len(values),
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "stdev": statistics.stdev(values) if len(values) > 1 else 0,
            "min": min(values),
            "max": max(values),
            "recent_avg": statistics.mean(values[-10:]) if len(values) >= 10 else statistics.mean(values)
        }