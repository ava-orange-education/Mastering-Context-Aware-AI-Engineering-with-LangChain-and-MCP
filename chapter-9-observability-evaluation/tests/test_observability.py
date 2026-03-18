"""
Tests for observability components.
"""

import pytest
from observability.trace_logger import TraceLogger
from observability.prometheus_exporter import PrometheusExporter
from monitoring.agent_monitor import AgentMonitor
from monitoring.performance_tracker import PerformanceTracker
from monitoring.error_tracker import ErrorTracker


class TestTraceLogger:
    def test_start_and_end_trace(self):
        """Test trace lifecycle"""
        logger = TraceLogger()
        
        trace_id = logger.start_trace("test_agent", "test query")
        
        assert trace_id is not None
        assert trace_id in logger.traces
        
        logger.log_step(trace_id, "thought", "Thinking...")
        logger.end_trace(trace_id, response="Response", status="completed")
        
        trace = logger.get_trace(trace_id)
        
        assert trace is not None
        assert trace.status == "completed"
        assert len(trace.steps) == 1


class TestPrometheusExporter:
    def test_record_metrics(self):
        """Test metric recording"""
        exporter = PrometheusExporter()
        
        exporter.record_agent_request("test_agent", 1.5, "success")
        exporter.record_llm_call("claude-3-5-sonnet", 2.0, 100, 50, "success")
        exporter.record_retrieval(0.5, 5, "success")
        
        metrics = exporter.get_metrics_text()
        
        assert 'agent_requests_total' in metrics
        assert 'llm_calls_total' in metrics
        assert 'retrieval_requests_total' in metrics


class TestAgentMonitor:
    def test_record_execution(self):
        """Test execution recording"""
        monitor = AgentMonitor("test_agent")
        
        monitor.record_execution("query", "response", 1.0, success=True)
        monitor.record_execution("query2", "response2", 2.0, success=False)
        
        metrics = monitor.get_metrics()
        
        assert metrics['total_executions'] == 2
        assert metrics['successful_executions'] == 1
        assert metrics['failed_executions'] == 1
        assert metrics['success_rate'] == 0.5


class TestPerformanceTracker:
    def test_track_latency(self):
        """Test latency tracking"""
        tracker = PerformanceTracker()
        
        tracker.record_latency(1.0)
        tracker.record_latency(1.5)
        tracker.record_latency(2.0)
        
        stats = tracker.get_latency_stats()
        
        assert 'mean' in stats
        assert 'median' in stats
        assert stats['sample_count'] == 3


class TestErrorTracker:
    def test_log_error(self):
        """Test error logging"""
        tracker = ErrorTracker()
        
        tracker.log_error("timeout", "Request timeout", "api", "error")
        tracker.log_error("validation", "Invalid input", "api", "warning")
        
        assert tracker.get_error_count() == 2
        assert tracker.get_error_count(error_type="timeout") == 1
        
        by_type = tracker.get_errors_by_type()
        
        assert by_type['timeout'] == 1
        assert by_type['validation'] == 1