"""
Example 3: Prometheus Monitoring
Demonstrates Prometheus metrics export.
"""

import sys
sys.path.append('..')

from observability.prometheus_exporter import PrometheusExporter, PrometheusServer
import time
import random


def main():
    print("=" * 70)
    print("  PROMETHEUS MONITORING EXAMPLE")
    print("=" * 70 + "\n")
    
    # Initialize Prometheus exporter
    exporter = PrometheusExporter()
    
    print("1. RECORDING METRICS")
    print("-" * 70)
    
    # Simulate agent requests
    print("\nSimulating agent requests...")
    
    for i in range(10):
        agent_name = random.choice(["research_agent", "analysis_agent", "writing_agent"])
        duration = random.uniform(0.5, 3.0)
        status = "success" if random.random() > 0.1 else "failure"
        
        exporter.record_agent_request(agent_name, duration, status)
        
        if status == "failure":
            exporter.record_agent_error(agent_name, "timeout")
    
    print("✓ Recorded 10 agent requests")
    
    # Simulate LLM calls
    print("\nSimulating LLM calls...")
    
    for i in range(5):
        model = "claude-3-5-sonnet-20241022"
        duration = random.uniform(1.0, 2.5)
        input_tokens = random.randint(100, 500)
        output_tokens = random.randint(50, 300)
        
        exporter.record_llm_call(model, duration, input_tokens, output_tokens)
    
    print("✓ Recorded 5 LLM calls")
    
    # Simulate retrievals
    print("\nSimulating retrievals...")
    
    for i in range(8):
        duration = random.uniform(0.1, 0.5)
        doc_count = random.randint(3, 10)
        
        exporter.record_retrieval(duration, doc_count)
    
    print("✓ Recorded 8 retrievals")
    
    # Record evaluation scores
    print("\nRecording evaluation scores...")
    
    exporter.record_evaluation_score("relevance", 0.85)
    exporter.record_evaluation_score("coherence", 0.92)
    exporter.record_evaluation_score("groundedness", 0.78)
    exporter.record_evaluation_score("factuality", 0.88)
    
    print("✓ Recorded evaluation scores")
    
    # Set active agents
    exporter.set_active_agents(3)
    
    print("\n2. VIEWING METRICS")
    print("-" * 70)
    
    # Get metrics in Prometheus format
    metrics_text = exporter.get_metrics_text()
    
    print("\nSample metrics (first 10 lines):")
    print("-" * 70)
    
    lines = metrics_text.split('\n')[:15]
    for line in lines:
        if line and not line.startswith('#'):
            print(line)
    
    print("\n3. STARTING METRICS SERVER")
    print("-" * 70)
    
    try:
        server = PrometheusServer(exporter, port=8000)
        print("\n✓ Prometheus metrics server would start on port 8000")
        print("  Metrics available at: http://localhost:8000/metrics")
        print("\n  To scrape with Prometheus, add to prometheus.yml:")
        print("  scrape_configs:")
        print("    - job_name: 'ai-agent'")
        print("      static_configs:")
        print("        - targets: ['localhost:8000']")
        
        # Don't actually start server in example
        # server.start()
        
    except Exception as e:
        print(f"  Note: Server not started in example mode")
    
    print("\n" + "=" * 70)
    print("  PROMETHEUS MONITORING DEMONSTRATION COMPLETE")
    print("=" * 70)
    
    print("\n💡 Metric types recorded:")
    print("   • agent_requests_total - Total requests by agent and status")
    print("   • agent_request_duration_seconds - Request latency histogram")
    print("   • agent_errors_total - Errors by agent and type")
    print("   • llm_calls_total - LLM API calls")
    print("   • llm_tokens_total - Token usage")
    print("   • llm_latency_seconds - LLM latency")
    print("   • retrieval_requests_total - Retrieval operations")
    print("   • evaluation_score - Latest evaluation scores")


if __name__ == "__main__":
    main()