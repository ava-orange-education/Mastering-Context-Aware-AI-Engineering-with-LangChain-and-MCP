"""
Example 6: Dashboard Creation
Demonstrates building evaluation dashboards.
"""

import sys
sys.path.append('..')

from visualization.dashboard_builder import DashboardBuilder
from visualization.grafana_config import GrafanaConfig
import json


def main():
    print("=" * 70)
    print("  DASHBOARD CREATION EXAMPLE")
    print("=" * 70 + "\n")
    
    # Initialize dashboard builder
    builder = DashboardBuilder()
    
    print("1. CREATING EVALUATION DASHBOARD")
    print("-" * 70)
    
    # Create standard evaluation dashboard
    eval_dashboard = builder.create_evaluation_dashboard()
    
    print(f"✓ Dashboard created: {eval_dashboard['title']}")
    print(f"  Panels: {len(eval_dashboard['panels'])}")
    print(f"  Refresh interval: {eval_dashboard['refresh_interval']}s")
    
    print("\nPanels:")
    for panel in eval_dashboard['panels']:
        print(f"  • {panel['title']} ({panel['type']})")
    
    # Create performance dashboard
    print("\n2. CREATING PERFORMANCE DASHBOARD")
    print("-" * 70)
    
    perf_dashboard = builder.create_performance_dashboard()
    
    print(f"✓ Dashboard created: {perf_dashboard['title']}")
    print(f"  Panels: {len(perf_dashboard['panels'])}")
    
    print("\nPanels:")
    for panel in perf_dashboard['panels']:
        print(f"  • {panel['title']} ({panel['type']})")
    
    # Create custom dashboard
    print("\n3. CREATING CUSTOM DASHBOARD")
    print("-" * 70)
    
    custom = builder.create_dashboard(
        name="agent_metrics",
        title="Agent Metrics Dashboard",
        description="Custom metrics for AI agents"
    )
    
    # Add custom panels
    builder.add_metric_panel(
        "agent_metrics",
        "success_rate",
        "Success Rate",
        panel_type="gauge",
        thresholds={'good': 0.95, 'warning': 0.85, 'critical': 0.75}
    )
    
    builder.add_time_series_panel(
        "agent_metrics",
        ["latency", "throughput", "error_rate"],
        "Performance Trends"
    )
    
    builder.add_comparison_panel(
        "agent_metrics",
        ["research_agent", "analysis_agent", "writing_agent"],
        "success_rate",
        "Agent Comparison"
    )
    
    builder.add_table_panel(
        "agent_metrics",
        ["Agent", "Queries", "Success Rate", "Avg Latency"],
        "agent_stats",
        "Agent Statistics"
    )
    
    print(f"✓ Custom dashboard created with {len(custom['panels'])} panels")
    
    # Export dashboards
    print("\n4. EXPORTING DASHBOARDS")
    print("-" * 70)
    
    builder.export_dashboard("evaluation_dashboard", "eval_dashboard.json")
    print("✓ Exported: eval_dashboard.json")
    
    builder.export_dashboard("performance_dashboard", "perf_dashboard.json")
    print("✓ Exported: perf_dashboard.json")
    
    builder.export_dashboard("agent_metrics", "agent_dashboard.json")
    print("✓ Exported: agent_dashboard.json")
    
    # Create Grafana dashboards
    print("\n5. CREATING GRAFANA DASHBOARDS")
    print("-" * 70)
    
    grafana = GrafanaConfig()
    
    # Evaluation dashboard for Grafana
    grafana_eval = grafana.create_evaluation_dashboard()
    
    with open("grafana_evaluation.json", "w") as f:
        json.dump(grafana_eval, f, indent=2)
    
    print("✓ Grafana evaluation dashboard: grafana_evaluation.json")
    
    # Performance dashboard for Grafana
    grafana_perf = grafana.create_performance_dashboard()
    
    with open("grafana_performance.json", "w") as f:
        json.dump(grafana_perf, f, indent=2)
    
    print("✓ Grafana performance dashboard: grafana_performance.json")
    
    print("\n6. DASHBOARD CONFIGURATION PREVIEW")
    print("-" * 70)
    
    # Show sample configuration
    sample_panel = eval_dashboard['panels'][0]
    
    print("\nSample Panel Configuration:")
    print(f"  Title: {sample_panel['title']}")
    print(f"  Type: {sample_panel['type']}")
    print(f"  Metric: {sample_panel['metric']}")
    print(f"  Thresholds:")
    for key, value in sample_panel['thresholds'].items():
        print(f"    {key}: {value}")
    
    print("\n" + "=" * 70)
    print("  DASHBOARD CREATION DEMONSTRATION COMPLETE")
    print("=" * 70)
    
    print("\n💡 Dashboard Usage:")
    print("   • Use eval_dashboard.json for evaluation metrics")
    print("   • Use perf_dashboard.json for performance monitoring")
    print("   • Import grafana_*.json into Grafana")
    print("   • Customize panels based on your metrics")
    print("\n💡 Next Steps:")
    print("   1. Set up Prometheus to scrape metrics")
    print("   2. Import Grafana dashboards")
    print("   3. Configure alerting rules")
    print("   4. Customize visualizations")


if __name__ == "__main__":
    main()