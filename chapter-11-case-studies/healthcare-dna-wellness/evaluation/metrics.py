"""
Healthcare Metrics

Metrics for evaluating healthcare AI system performance
"""

from typing import Dict, Any, List
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class HealthcareMetrics:
    """
    Metrics collector for healthcare AI system
    """
    
    def __init__(self):
        self.metrics_store: List[Dict[str, Any]] = []
    
    def record_recommendation(
        self,
        recommendation_id: str,
        patient_id: str,
        processing_time: float,
        accuracy_score: float,
        safety_score: float,
        confidence_score: float
    ) -> None:
        """
        Record recommendation metrics
        
        Args:
            recommendation_id: Unique recommendation ID
            patient_id: Patient ID
            processing_time: Processing time in seconds
            accuracy_score: Medical accuracy score
            safety_score: Safety score
            confidence_score: Confidence score
        """
        
        metric = {
            "timestamp": datetime.utcnow(),
            "recommendation_id": recommendation_id,
            "patient_id": patient_id,
            "processing_time": processing_time,
            "accuracy_score": accuracy_score,
            "safety_score": safety_score,
            "confidence_score": confidence_score
        }
        
        self.metrics_store.append(metric)
    
    def get_summary_metrics(
        self,
        days: int = 7
    ) -> Dict[str, Any]:
        """
        Get summary metrics for time period
        
        Args:
            days: Number of days to analyze
        
        Returns:
            Summary metrics
        """
        
        cutoff = datetime.utcnow() - timedelta(days=days)
        
        recent_metrics = [
            m for m in self.metrics_store
            if m["timestamp"] > cutoff
        ]
        
        if not recent_metrics:
            return {
                "total_recommendations": 0,
                "period_days": days
            }
        
        # Calculate averages
        avg_processing_time = sum(m["processing_time"] for m in recent_metrics) / len(recent_metrics)
        avg_accuracy = sum(m["accuracy_score"] for m in recent_metrics) / len(recent_metrics)
        avg_safety = sum(m["safety_score"] for m in recent_metrics) / len(recent_metrics)
        avg_confidence = sum(m["confidence_score"] for m in recent_metrics) / len(recent_metrics)
        
        # Calculate percentiles
        processing_times = sorted([m["processing_time"] for m in recent_metrics])
        p95_processing = processing_times[int(len(processing_times) * 0.95)]
        
        return {
            "total_recommendations": len(recent_metrics),
            "period_days": days,
            "avg_processing_time": round(avg_processing_time, 2),
            "p95_processing_time": round(p95_processing, 2),
            "avg_accuracy_score": round(avg_accuracy, 3),
            "avg_safety_score": round(avg_safety, 3),
            "avg_confidence_score": round(avg_confidence, 3),
            "high_accuracy_rate": sum(1 for m in recent_metrics if m["accuracy_score"] >= 0.9) / len(recent_metrics),
            "safe_recommendations_rate": sum(1 for m in recent_metrics if m["safety_score"] >= 0.95) / len(recent_metrics)
        }
    
    def get_quality_metrics(self) -> Dict[str, float]:
        """Get quality metrics"""
        
        if not self.metrics_store:
            return {}
        
        recent = self.metrics_store[-100:]  # Last 100
        
        return {
            "accuracy_pass_rate": sum(1 for m in recent if m["accuracy_score"] >= 0.8) / len(recent),
            "safety_pass_rate": sum(1 for m in recent if m["safety_score"] >= 0.95) / len(recent),
            "high_confidence_rate": sum(1 for m in recent if m["confidence_score"] >= 0.8) / len(recent)
        }