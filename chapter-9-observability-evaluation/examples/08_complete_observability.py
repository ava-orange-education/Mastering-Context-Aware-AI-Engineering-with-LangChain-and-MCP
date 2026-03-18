"""
Example 8: Complete Observability System
Demonstrates full observability stack integration.
"""

import sys
sys.path.append('..')

from observability.langfuse_integration import LangfuseIntegration
from observability.langsmith_integration import LangSmithIntegration
from observability.prometheus_exporter import PrometheusExporter
from observability.trace_logger import TraceLogger
from monitoring.agent_monitor import MultiAgentMonitor
from monitoring.performance_tracker import PerformanceTracker
from monitoring.error_tracker import ErrorTracker
from monitoring.alert_manager import AlertManager, create_default_rules
from evaluation.evaluator import Evaluator
import time
import random
import os


class ObservableAgent:
    """Agent with comprehensive observability"""
    
    def __init__(self, name, langfuse, langsmith, prometheus, trace_logger, monitor):
        self.name = name
        self.langfuse = langfuse
        self.langsmith = langsmith
        self.prometheus = prometheus
        self.trace_logger = trace_logger
        self.monitor = monitor.get_monitor(name)
    
    def run(self, query):
        # Start trace
        trace_id = self.trace_logger.start_trace(self.name, query)
        start_time = time.time()
        
        try:
            # Log thinking step
            self.trace_logger.log_step(trace_id, "thought", f"Analyzing query: {query}")
            
            # Simulate processing
            time.sleep(random.uniform(0.1, 0.5))
            
            # Log action
            self.trace_logger.log_step(trace_id, "action", "Retrieving information")
            
            # Generate response
            response = f"Response from {self.name}: Processed '{query}'"
            
            # Log observation
            self.trace_logger.log_step(trace_id, "observation", f"Generated response: {response[:50]}...")
            
            # Calculate duration
            duration = time.time() - start_time
            
            # End trace
            self.trace_logger.end_trace(trace_id, response=response, status="completed")
            
            # Record metrics
            self.monitor.record_execution(query, response, duration, success=True)
            self.prometheus.record_agent_request(self.name, duration, "success")
            
            # Trace in Langfuse
            if self.langfuse.enabled:
                self.langfuse.trace_agent_run(self.name, query, response)
            
            # Log in LangSmith
            if self.langsmith.enabled:
                self.langsmith.log_agent_run(self.name, query, response)
            
            return {'result': response, 'trace_id': trace_id}
            
        except Exception as e:
            duration = time.time() - start_time
            
            # Log error
            self.trace_logger.end_trace(trace_id, status="failed", error=str(e))
            self.monitor.record_error(query, str(e), "processing_error")
            self.prometheus.record_agent_error(self.name, "processing_error")
            
            raise


def main():
    print("=" * 70)
    print("  COMPLETE OBSERVABILITY SYSTEM")
    print("=" * 70 + "\n")
    
    # Initialize observability components
    print("1. INITIALIZING OBSERVABILITY STACK")
    print("-" * 70)
    
    langfuse = LangfuseIntegration(
        public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
        secret_key=os.getenv("LANGFUSE_SECRET_KEY")
    )
    print(f"✓ Langfuse: {'Enabled' if langfuse.enabled else 'Disabled (no credentials)'}")
    
    langsmith = LangSmithIntegration(
        api_key=os.getenv("LANGCHAIN_API_KEY")
    )
    print(f"✓ LangSmith: {'Enabled' if langsmith.enabled else 'Disabled (no API key)'}")
    
    prometheus = PrometheusExporter()
    print("✓ Prometheus: Enabled")
    
    trace_logger = TraceLogger(log_file="traces.jsonl")
    print("✓ Trace Logger: Enabled")
    
    multi_monitor = MultiAgentMonitor()
    print("✓ Agent Monitor: Enabled")
    
    perf_tracker = PerformanceTracker(window_minutes=60)
    print("✓ Performance Tracker: Enabled")
    
    error_tracker = ErrorTracker()
    print("✓ Error Tracker: Enabled")
    
    alert_manager = AlertManager()
    create_default_rules(alert_manager)
    print("✓ Alert Manager: Enabled with default rules")
    
    evaluator = Evaluator()
    print("✓ Evaluator: Enabled")
    
    # Create observable agents
    print("\n2. CREATING OBSERVABLE AGENTS")
    print("-" * 70)
    
    agents = {
        'research': ObservableAgent(
            "research_agent", langfuse, langsmith, prometheus, 
            trace_logger, multi_monitor
        ),
        'analysis': ObservableAgent(
            "analysis_agent", langfuse, langsmith, prometheus,
            trace_logger, multi_monitor
        ),
        'synthesis': ObservableAgent(
            "synthesis_agent", langfuse, langsmith, prometheus,
            trace_logger, multi_monitor
        )
    }
    
    for name in agents.keys():
        print(f"  ✓ {name}_agent created with full observability")
    
    # Simulate agent execution
    print("\n3. SIMULATING AGENT EXECUTION")
    print("-" * 70)
    
    queries = [
        "What is machine learning?",
        "Explain neural networks",
        "Compare supervised and unsupervised learning",
        "What are the applications of AI?",
        "Describe deep learning"
    ]
    
    print(f"\nExecuting {len(queries)} queries across {len(agents)} agents...\n")
    
    for query in queries:
        agent_name = random.choice(list(agents.keys()))
        agent = agents[agent_name]
        
        print(f"  {agent_name}_agent: {query[:40]}...")
        
        start = time.time()
        
        try:
            result = agent.run(query)
            latency = time.time() - start
            
            # Track performance
            perf_tracker.record_latency(latency, agent.name)
            perf_tracker.record_throughput(1, agent.name)
            
            # Evaluate response
            eval_result = evaluator.evaluate_response(
                query=query,
                response=result['result']
            )
            
            # Record evaluation scores in Prometheus
            prometheus.record_evaluation_score("relevance", eval_result['relevance'])
            prometheus.record_evaluation_score("coherence", eval_result['coherence'])
            
            print(f"    ✓ Success (score: {eval_result['overall_score']:.2f}, latency: {latency:.3f}s)")
            
        except Exception as e:
            error_tracker.log_error(
                error_type="execution_error",
                error_message=str(e),
                component=agent.name,
                severity="error"
            )
            print(f"    ✗ Failed: {e}")
        
        time.sleep(0.1)  # Small delay between requests
    
    # Display monitoring data
    print("\n4. MONITORING DASHBOARD")
    print("-" * 70)
    
    # Agent metrics
    summary = multi_monitor.get_summary()
    
    print(f"\nAgent Summary:")
    print(f"  Total Agents: {summary['total_agents']}")
    print(f"  Total Executions: {summary['total_executions']}")
    print(f"  Success Rate: {summary['overall_success_rate']:.1%}")
    
    print("\nPer-Agent Metrics:")
    for agent_name, metrics in summary['agent_metrics'].items():
        print(f"\n  {agent_name}:")
        print(f"    Executions: {metrics['total_executions']}")
        print(f"    Success Rate: {metrics['success_rate']:.1%}")
        print(f"    Avg Latency: {metrics['average_latency']:.3f}s")
    
    # Performance metrics
    print("\n5. PERFORMANCE METRICS")
    print("-" * 70)
    
    latency_stats = perf_tracker.get_latency_stats()
    
    if latency_stats:
        print(f"\nLatency Statistics:")
        print(f"  Mean: {latency_stats['mean']:.3f}s")
        print(f"  Median: {latency_stats['median']:.3f}s")
        print(f"  P95: {latency_stats['p95']:.3f}s")
        print(f"  P99: {latency_stats['p99']:.3f}s")
        print(f"  Min: {latency_stats['min']:.3f}s")
        print(f"  Max: {latency_stats['max']:.3f}s")
    
    # Error tracking
    print("\n6. ERROR TRACKING")
    print("-" * 70)
    
    error_summary = error_tracker.get_summary()
    
    print(f"\nError Summary:")
    print(f"  Total Errors: {error_summary['total_errors']}")
    print(f"  Errors (last hour): {error_summary['errors_last_hour']}")
    
    if error_summary['errors_by_type']:
        print(f"\nErrors by Type:")
        for error_type, count in error_summary['errors_by_type'].items():
            print(f"    {error_type}: {count}")
    
    # Check alerts
    print("\n7. ALERT STATUS")
    print("-" * 70)
    
    # Simulate checking metrics against rules
    current_metrics = {
        'error_rate': error_summary['errors_last_hour'] / max(summary['total_executions'], 1),
        'avg_latency': latency_stats.get('mean', 0),
        'throughput': summary['total_executions']
    }
    
    alert_manager.check_rules(current_metrics)
    
    active_alerts = alert_manager.get_active_alerts()
    
    if active_alerts:
        print(f"\n⚠️  Active Alerts: {len(active_alerts)}")
        for alert in active_alerts:
            print(f"  [{alert.severity.upper()}] {alert.message}")
    else:
        print("\n✓ No active alerts")
    
    # Trace analysis
    print("\n8. TRACE ANALYSIS")
    print("-" * 70)
    
    recent_traces = trace_logger.get_recent_traces(count=3)
    
    print(f"\nRecent Traces ({len(recent_traces)}):")
    for trace in recent_traces:
        summary = trace_logger.get_trace_summary(trace.trace_id)
        if summary:
            print(f"\n  {summary['trace_id']}:")
            print(f"    Agent: {summary['agent_name']}")
            print(f"    Status: {summary['status']}")
            print(f"    Steps: {summary['step_count']}")
            print(f"    Duration: {summary['total_duration_ms']:.0f}ms")
    
    # Export traces
    trace_logger.export_traces("traces_export.json")
    print(f"\n✓ Traces exported to traces_export.json")
    
    # Prometheus metrics
    print("\n9. PROMETHEUS METRICS")
    print("-" * 70)
    
    metrics_sample = prometheus.get_metrics_text().split('\n')[:10]
    print("\nSample Metrics:")
    for line in metrics_sample:
        if line and not line.startswith('#'):
            print(f"  {line}")
    
    print(f"\n✓ Full metrics available at /metrics endpoint")
    
    # Flush external integrations
    print("\n10. FLUSHING TRACES")
    print("-" * 70)
    
    if langfuse.enabled:
        langfuse.flush()
        print("✓ Langfuse traces flushed")
    
    print("\n" + "=" * 70)
    print("  COMPLETE OBSERVABILITY DEMONSTRATION COMPLETE")
    print("=" * 70)
    
    print("\n📊 Observability Stack Summary:")
    print("   ✓ Langfuse - Trace visualization and analysis")
    print("   ✓ LangSmith - LLM monitoring and debugging")
    print("   ✓ Prometheus - Metrics collection")
    print("   ✓ Trace Logger - Detailed execution traces")
    print("   ✓ Agent Monitor - Per-agent metrics")
    print("   ✓ Performance Tracker - Latency and throughput")
    print("   ✓ Error Tracker - Error aggregation")
    print("   ✓ Alert Manager - Real-time alerting")
    
    print("\n💡 Next Steps:")
    print("   1. Set up Grafana dashboards for metrics visualization")
    print("   2. Configure Langfuse/LangSmith with your API keys")
    print("   3. Set up alerting channels (email, Slack, PagerDuty)")
    print("   4. Implement custom metrics for your use case")
    print("   5. Create runbooks for common alerts")


if __name__ == "__main__":
    main()