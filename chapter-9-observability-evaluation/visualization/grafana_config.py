"""
Generate Grafana dashboard configurations.
"""

from typing import Dict, Any, List, Optional
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GrafanaConfig:
    """Generate Grafana dashboard configurations"""
    
    def __init__(self):
        """Initialize Grafana config generator"""
        self.datasource = "Prometheus"
    
    def create_dashboard(self, title: str, uid: str,
                        description: str = "") -> Dict[str, Any]:
        """
        Create Grafana dashboard structure
        
        Args:
            title: Dashboard title
            uid: Unique dashboard ID
            description: Dashboard description
            
        Returns:
            Grafana dashboard JSON
        """
        dashboard = {
            "dashboard": {
                "title": title,
                "uid": uid,
                "tags": ["ai-agent", "evaluation"],
                "timezone": "browser",
                "schemaVersion": 16,
                "version": 0,
                "refresh": "30s",
                "description": description,
                "panels": []
            },
            "overwrite": True
        }
        
        return dashboard
    
    def create_gauge_panel(self, title: str, metric: str,
                          panel_id: int, x: int, y: int,
                          w: int = 6, h: int = 8) -> Dict[str, Any]:
        """
        Create gauge panel
        
        Args:
            title: Panel title
            metric: Metric name
            panel_id: Panel ID
            x, y: Position
            w, h: Width and height
            
        Returns:
            Panel configuration
        """
        return {
            "id": panel_id,
            "title": title,
            "type": "gauge",
            "datasource": self.datasource,
            "gridPos": {"x": x, "y": y, "w": w, "h": h},
            "targets": [{
                "expr": metric,
                "refId": "A"
            }],
            "options": {
                "showThresholdLabels": False,
                "showThresholdMarkers": True
            },
            "fieldConfig": {
                "defaults": {
                    "thresholds": {
                        "mode": "absolute",
                        "steps": [
                            {"value": 0, "color": "red"},
                            {"value": 0.6, "color": "yellow"},
                            {"value": 0.8, "color": "green"}
                        ]
                    },
                    "min": 0,
                    "max": 1
                }
            }
        }
    
    def create_graph_panel(self, title: str, metrics: List[str],
                          panel_id: int, x: int, y: int,
                          w: int = 12, h: int = 8) -> Dict[str, Any]:
        """
        Create time series graph panel
        
        Args:
            title: Panel title
            metrics: List of metrics to plot
            panel_id: Panel ID
            x, y: Position
            w, h: Width and height
            
        Returns:
            Panel configuration
        """
        targets = []
        for idx, metric in enumerate(metrics):
            targets.append({
                "expr": metric,
                "refId": chr(65 + idx),  # A, B, C, ...
                "legendFormat": metric
            })
        
        return {
            "id": panel_id,
            "title": title,
            "type": "graph",
            "datasource": self.datasource,
            "gridPos": {"x": x, "y": y, "w": w, "h": h},
            "targets": targets,
            "yaxes": [
                {"format": "short", "label": None, "logBase": 1},
                {"format": "short", "label": None, "logBase": 1}
            ],
            "lines": True,
            "fill": 1,
            "linewidth": 2
        }
    
    def create_stat_panel(self, title: str, metric: str,
                         panel_id: int, x: int, y: int,
                         w: int = 4, h: int = 4) -> Dict[str, Any]:
        """
        Create stat panel (single number)
        
        Args:
            title: Panel title
            metric: Metric name
            panel_id: Panel ID
            x, y: Position
            w, h: Width and height
            
        Returns:
            Panel configuration
        """
        return {
            "id": panel_id,
            "title": title,
            "type": "stat",
            "datasource": self.datasource,
            "gridPos": {"x": x, "y": y, "w": w, "h": h},
            "targets": [{
                "expr": metric,
                "refId": "A"
            }],
            "options": {
                "graphMode": "area",
                "colorMode": "value",
                "justifyMode": "auto",
                "textMode": "auto"
            }
        }
    
    def create_evaluation_dashboard(self) -> Dict[str, Any]:
        """
        Create standard evaluation dashboard for Grafana
        
        Returns:
            Grafana dashboard JSON
        """
        dashboard = self.create_dashboard(
            title="AI Agent Evaluation",
            uid="ai-agent-eval",
            description="Comprehensive evaluation metrics for AI agents"
        )
        
        panels = []
        
        # Row 1: Metric gauges
        panels.append(self.create_gauge_panel(
            "Relevance", "evaluation_score{metric_name=\"relevance\"}",
            1, 0, 0, 6, 8
        ))
        
        panels.append(self.create_gauge_panel(
            "Coherence", "evaluation_score{metric_name=\"coherence\"}",
            2, 6, 0, 6, 8
        ))
        
        panels.append(self.create_gauge_panel(
            "Groundedness", "evaluation_score{metric_name=\"groundedness\"}",
            3, 12, 0, 6, 8
        ))
        
        panels.append(self.create_gauge_panel(
            "Factuality", "evaluation_score{metric_name=\"factuality\"}",
            4, 18, 0, 6, 8
        ))
        
        # Row 2: Time series
        panels.append(self.create_graph_panel(
            "Evaluation Metrics Over Time",
            [
                "evaluation_score{metric_name=\"relevance\"}",
                "evaluation_score{metric_name=\"coherence\"}",
                "evaluation_score{metric_name=\"groundedness\"}",
                "evaluation_score{metric_name=\"factuality\"}"
            ],
            5, 0, 8, 24, 8
        ))
        
        # Row 3: Agent metrics
        panels.append(self.create_stat_panel(
            "Total Requests", "agent_requests_total",
            6, 0, 16, 6, 4
        ))
        
        panels.append(self.create_stat_panel(
            "Success Rate", 
            "agent_requests_total{status=\"success\"} / agent_requests_total",
            7, 6, 16, 6, 4
        ))
        
        panels.append(self.create_stat_panel(
            "Avg Latency", "rate(agent_request_duration_seconds_sum[5m]) / rate(agent_request_duration_seconds_count[5m])",
            8, 12, 16, 6, 4
        ))
        
        panels.append(self.create_stat_panel(
            "Total Errors", "agent_errors_total",
            9, 18, 16, 6, 4
        ))
        
        dashboard["dashboard"]["panels"] = panels
        
        logger.info("Grafana evaluation dashboard created")
        
        return dashboard
    
    def create_performance_dashboard(self) -> Dict[str, Any]:
        """
        Create performance monitoring dashboard
        
        Returns:
            Grafana dashboard JSON
        """
        dashboard = self.create_dashboard(
            title="Agent Performance",
            uid="agent-performance",
            description="Performance monitoring for AI agents"
        )
        
        panels = []
        
        # Latency graph
        panels.append(self.create_graph_panel(
            "Request Latency",
            [
                "histogram_quantile(0.50, rate(agent_request_duration_seconds_bucket[5m]))",
                "histogram_quantile(0.95, rate(agent_request_duration_seconds_bucket[5m]))",
                "histogram_quantile(0.99, rate(agent_request_duration_seconds_bucket[5m]))"
            ],
            1, 0, 0, 12, 8
        ))
        
        # Throughput graph
        panels.append(self.create_graph_panel(
            "Request Rate",
            ["rate(agent_requests_total[5m])"],
            2, 12, 0, 12, 8
        ))
        
        # LLM metrics
        panels.append(self.create_graph_panel(
            "LLM Latency",
            ["rate(llm_latency_seconds_sum[5m]) / rate(llm_latency_seconds_count[5m])"],
            3, 0, 8, 12, 8
        ))
        
        panels.append(self.create_graph_panel(
            "Token Usage",
            [
                "rate(llm_tokens_total{type=\"input\"}[5m])",
                "rate(llm_tokens_total{type=\"output\"}[5m])"
            ],
            4, 12, 8, 12, 8
        ))
        
        dashboard["dashboard"]["panels"] = panels
        
        logger.info("Grafana performance dashboard created")
        
        return dashboard
    
    def save_dashboard(self, dashboard: Dict[str, Any], filepath: str):
        """
        Save dashboard to JSON file
        
        Args:
            dashboard: Dashboard configuration
            filepath: Output file path
        """
        with open(filepath, 'w') as f:
            json.dump(dashboard, f, indent=2)
        
        logger.info(f"Grafana dashboard saved: {filepath}")