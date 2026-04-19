"""
Example 1: Anomaly Detection

Demonstrates anomaly detection in metrics
"""

import asyncio
import sys
sys.path.append('..')

from agents.anomaly_detection_agent import AnomalyDetectionAgent
from monitoring.anomaly_detector import AnomalyDetector
from monitoring.metric_aggregator import MetricAggregator
from datetime import datetime
import random


async def main():
    """Run anomaly detection example"""
    
    print("="*70)
    print("DevOps Monitoring - Anomaly Detection Example")
    print("="*70)
    print()
    
    # Initialize components
    print("Initializing components...")
    agent = AnomalyDetectionAgent()
    detector = AnomalyDetector()
    aggregator = MetricAggregator()
    
    # Simulate baseline metrics (normal operation)
    print("\n1. Building baseline from normal metrics...")
    baseline_metrics = {
        "cpu_usage": 0.65,
        "memory_usage": 0.70,
        "request_rate": 100.0,
        "error_rate": 0.01,
        "response_time": 150.0
    }
    
    # Add baseline data
    for i in range(50):
        # Add some variation
        for metric_name, base_value in baseline_metrics.items():
            variation = random.uniform(-0.05, 0.05) * base_value
            value = base_value + variation
            detector.add_datapoint(metric_name, value)
            aggregator.add_metric(metric_name, value)
    
    print("   Baseline established from 50 data points per metric")
    
    # Get baseline statistics
    print("\n2. Baseline Statistics:")
    for metric_name in baseline_metrics:
        stats = detector.get_baseline_stats(metric_name)
        if stats:
            print(f"   {metric_name}:")
            print(f"     Mean: {stats['mean']:.2f}")
            print(f"     Std Dev: {stats['stdev']:.2f}")
            print(f"     Range: [{stats['min']:.2f}, {stats['max']:.2f}]")
    
    # Simulate anomalous metrics
    print("\n3. Introducing anomalous metrics...")
    anomalous_metrics = {
        "cpu_usage": 0.95,          # Spike (normal: 0.65)
        "memory_usage": 0.72,       # Normal
        "request_rate": 450.0,      # Major spike (normal: 100)
        "error_rate": 0.15,         # High errors (normal: 0.01)
        "response_time": 850.0      # High latency (normal: 150)
    }
    
    print("\n   Current Metrics:")
    for metric_name, value in anomalous_metrics.items():
        baseline_value = baseline_metrics[metric_name]
        change_pct = ((value - baseline_value) / baseline_value) * 100
        print(f"   {metric_name}: {value:.2f} ({change_pct:+.1f}% change)")
    
    # Detect anomalies
    print("\n4. Running anomaly detection...")
    anomalies = detector.detect_anomalies(anomalous_metrics)
    
    if anomalies:
        print(f"\n   ⚠️  {len(anomalies)} anomalies detected!")
        print()
        for i, anomaly in enumerate(anomalies, 1):
            print(f"   Anomaly {i}:")
            print(f"     Metric: {anomaly['metric']}")
            print(f"     Type: {anomaly['type']}")
            print(f"     Severity: {anomaly['severity']}")
            print(f"     Description: {anomaly['description']}")
            if 'z_score' in anomaly:
                print(f"     Z-Score: {anomaly['z_score']}")
            if 'percentage_change' in anomaly:
                print(f"     Change: {anomaly['percentage_change']:.1f}%")
            print()
    else:
        print("\n   ✓ No anomalies detected")
    
    # Run AI agent analysis
    print("5. Running AI agent analysis...")
    
    # Get baseline for context
    baseline_stats = {}
    for metric_name in baseline_metrics:
        stats = detector.get_baseline_stats(metric_name)
        if stats:
            baseline_stats[metric_name] = stats
    
    # Process with agent
    response = await agent.process({
        "metrics": anomalous_metrics,
        "baseline": baseline_stats,
        "context": {
            "time_of_day": "peak_hours",
            "recent_deployments": False,
            "load_pattern": "unexpected"
        }
    })
    
    print("\n   AI Analysis:")
    print("   " + "-"*60)
    # Print first 500 chars of analysis
    analysis_preview = response.content[:500]
    if len(response.content) > 500:
        analysis_preview += "..."
    print(f"   {analysis_preview}")
    print("   " + "-"*60)
    
    print(f"\n   Confidence: {response.confidence:.2%}")
    print(f"   Anomalies confirmed: {response.metadata.get('anomalies_detected')}")
    
    # Show trend analysis
    print("\n6. Trend Analysis:")
    for metric_name in ["cpu_usage", "request_rate", "error_rate"]:
        trend = aggregator.get_trend(metric_name, "15m")
        if trend:
            icon = "↗" if trend == "increasing" else "↘" if trend == "decreasing" else "→"
            print(f"   {metric_name}: {icon} {trend}")
    
    # Show recommendations
    print("\n7. Recommendations:")
    print("   • Investigate the sudden spike in request rate")
    print("   • Check for DDoS attack or legitimate traffic surge")
    print("   • Review error logs for error_rate increase")
    print("   • Consider scaling up resources if legitimate traffic")
    print("   • Set up alerts for similar patterns in the future")
    
    print("\n" + "="*70)
    print("Example completed!")
    print("="*70)


if __name__ == "__main__":
    asyncio.run(main())