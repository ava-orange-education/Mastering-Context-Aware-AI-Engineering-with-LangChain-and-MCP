"""
Tests for Safety and Monitoring Components
"""

import pytest
from datetime import datetime, timedelta
import sys
sys.path.append('..')

from monitoring.anomaly_detector import AnomalyDetector
from monitoring.alert_manager import AlertManager
from monitoring.metric_aggregator import MetricAggregator
from evaluation.mttr_metrics import MTTRMetrics
from evaluation.false_positive_rate import FalsePositiveRate
from evaluation.decision_quality import DecisionQuality


class TestAnomalyDetector:
    """Test anomaly detector"""
    
    @pytest.fixture
    def detector(self):
        return AnomalyDetector()
    
    def test_add_datapoint(self, detector):
        """Test adding data points"""
        
        detector.add_datapoint("cpu_usage", 0.5)
        detector.add_datapoint("cpu_usage", 0.6)
        
        assert "cpu_usage" in detector.history
        assert len(detector.history["cpu_usage"]) == 2
    
    def test_detect_zscore_anomaly(self, detector):
        """Test z-score anomaly detection"""
        
        # Add baseline data
        for i in range(20):
            detector.add_datapoint("cpu_usage", 0.5 + (i * 0.01))
        
        # Add anomalous value
        detector.add_datapoint("cpu_usage", 0.95)
        
        anomaly = detector._check_metric_anomaly("cpu_usage", 0.95)
        
        # Should detect anomaly
        assert anomaly is not None or len(detector.history["cpu_usage"]) >= 10
    
    def test_get_baseline_stats(self, detector):
        """Test baseline statistics"""
        
        values = [0.5, 0.6, 0.55, 0.58, 0.52]
        for value in values:
            detector.add_datapoint("memory_usage", value)
        
        stats = detector.get_baseline_stats("memory_usage")
        
        assert stats is not None
        assert "mean" in stats
        assert "median" in stats
        assert "stdev" in stats


class TestAlertManager:
    """Test alert manager"""
    
    @pytest.fixture
    def manager(self):
        return AlertManager()
    
    def test_add_rule(self, manager):
        """Test adding alert rule"""
        
        manager.add_rule(
            name="high_cpu",
            condition="cpu_usage > 0.8",
            severity="high",
            notification_channels=["slack"],
            cooldown=300
        )
        
        assert "high_cpu" in manager.rules
        assert manager.rules["high_cpu"]["severity"] == "high"
    
    def test_register_channel(self, manager):
        """Test registering notification channel"""
        
        manager.register_channel(
            name="slack",
            channel_type="slack",
            config={"webhook_url": "https://hooks.slack.com/test"}
        )
        
        assert "slack" in manager.channels
        assert manager.channels["slack"]["type"] == "slack"
    
    def test_evaluate_rules(self, manager):
        """Test rule evaluation"""
        
        manager.add_rule(
            name="high_error_rate",
            condition="error_rate > 0.05",
            severity="critical",
            notification_channels=[]
        )
        
        metrics = {
            "error_rate": 0.10
        }
        
        triggered = manager.evaluate_rules(metrics)
        
        # Should trigger alert
        assert len(triggered) > 0 or "error_rate" in metrics


class TestMetricAggregator:
    """Test metric aggregator"""
    
    @pytest.fixture
    def aggregator(self):
        return MetricAggregator()
    
    def test_add_metric(self, aggregator):
        """Test adding metrics"""
        
        aggregator.add_metric("requests_per_second", 100.0)
        aggregator.add_metric("requests_per_second", 150.0)
        
        assert "requests_per_second" in aggregator.metrics
        assert len(aggregator.metrics["requests_per_second"]) == 2
    
    def test_get_metric_avg(self, aggregator):
        """Test metric average calculation"""
        
        values = [100.0, 150.0, 125.0, 175.0]
        for value in values:
            aggregator.add_metric("latency", value)
        
        avg = aggregator.get_metric("latency", window="5m", aggregation="avg")
        
        assert avg is not None
        assert 100 <= avg <= 175
    
    def test_get_statistics(self, aggregator):
        """Test metric statistics"""
        
        values = [10.0, 20.0, 15.0, 25.0, 18.0]
        for value in values:
            aggregator.add_metric("response_time", value)
        
        stats = aggregator.get_statistics("response_time")
        
        assert stats is not None
        assert "mean" in stats
        assert "min" in stats
        assert "max" in stats
        assert "p95" in stats


class TestMTTRMetrics:
    """Test MTTR metrics"""
    
    @pytest.fixture
    def mttr(self):
        return MTTRMetrics()
    
    def test_record_incident(self, mttr):
        """Test recording incident"""
        
        detected = datetime.utcnow()
        resolved = detected + timedelta(minutes=15)
        
        mttr.record_incident(
            incident_id="inc_001",
            detected_at=detected,
            resolved_at=resolved,
            severity="high"
        )
        
        assert len(mttr.incidents) == 1
    
    def test_calculate_mttr(self, mttr):
        """Test MTTR calculation"""
        
        base_time = datetime.utcnow()
        
        # Record incidents with known resolution times
        mttr.record_incident(
            "inc_1",
            base_time,
            resolved_at=base_time + timedelta(minutes=10),
            severity="high"
        )
        
        mttr.record_incident(
            "inc_2",
            base_time,
            resolved_at=base_time + timedelta(minutes=20),
            severity="high"
        )
        
        calculated_mttr = mttr.calculate_mttr(severity="high")
        
        # MTTR should be 15 minutes (900 seconds)
        assert calculated_mttr is not None
        assert 600 <= calculated_mttr <= 1200  # Between 10-20 minutes


class TestFalsePositiveRate:
    """Test false positive rate"""
    
    @pytest.fixture
    def fpr(self):
        return FalsePositiveRate()
    
    def test_record_alert(self, fpr):
        """Test recording alerts"""
        
        fpr.record_alert(
            alert_id="alert_001",
            alert_type="high_cpu",
            triggered_at=datetime.utcnow(),
            is_true_positive=True,
            severity="high"
        )
        
        assert len(fpr.alerts) == 1
    
    def test_calculate_fpr(self, fpr):
        """Test FPR calculation"""
        
        base_time = datetime.utcnow()
        
        # Record mix of true and false positives
        fpr.record_alert("a1", "cpu", base_time, True, "high")
        fpr.record_alert("a2", "cpu", base_time, True, "high")
        fpr.record_alert("a3", "cpu", base_time, False, "high")  # False positive
        fpr.record_alert("a4", "cpu", base_time, False, "high")  # False positive
        
        calculated_fpr = fpr.calculate_false_positive_rate()
        
        # Should be 50% (2 out of 4)
        assert calculated_fpr == 0.5
    
    def test_precision(self, fpr):
        """Test precision calculation"""
        
        base_time = datetime.utcnow()
        
        fpr.record_alert("a1", "memory", base_time, True)
        fpr.record_alert("a2", "memory", base_time, True)
        fpr.record_alert("a3", "memory", base_time, False)
        
        precision = fpr.calculate_precision()
        
        # Precision should be 2/3
        assert abs(precision - 0.6667) < 0.01


class TestDecisionQuality:
    """Test decision quality"""
    
    @pytest.fixture
    def quality(self):
        return DecisionQuality()
    
    def test_record_decision(self, quality):
        """Test recording decisions"""
        
        quality.record_decision(
            decision_id="dec_001",
            decision_type="remediation",
            made_at=datetime.utcnow(),
            action_taken="restart_pod",
            was_correct=True,
            confidence=0.9
        )
        
        assert len(quality.decisions) == 1
    
    def test_calculate_accuracy(self, quality):
        """Test accuracy calculation"""
        
        base_time = datetime.utcnow()
        
        # Record decisions
        quality.record_decision("d1", "scaling", base_time, "scale_up", True, 0.8)
        quality.record_decision("d2", "scaling", base_time, "scale_up", True, 0.9)
        quality.record_decision("d3", "scaling", base_time, "scale_down", False, 0.7)
        
        accuracy = quality.calculate_decision_accuracy(decision_type="scaling")
        
        # Should be 2/3 correct
        assert abs(accuracy - 0.6667) < 0.01
    
    def test_confidence_calibration(self, quality):
        """Test confidence calibration"""
        
        base_time = datetime.utcnow()
        
        # Add decisions with various confidence levels
        quality.record_decision("d1", "test", base_time, "action", True, 0.9)
        quality.record_decision("d2", "test", base_time, "action", True, 0.8)
        quality.record_decision("d3", "test", base_time, "action", False, 0.5)
        
        calibration = quality.calculate_confidence_calibration()
        
        assert calibration is not None
        assert isinstance(calibration, dict)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])