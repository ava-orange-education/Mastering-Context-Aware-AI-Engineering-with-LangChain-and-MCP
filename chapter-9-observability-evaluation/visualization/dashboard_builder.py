"""
Build evaluation dashboards.
"""

from typing import Dict, Any, List, Optional
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DashboardBuilder:
    """Build interactive evaluation dashboards"""
    
    def __init__(self):
        """Initialize dashboard builder"""
        self.dashboards: Dict[str, Dict] = {}
    
    def create_dashboard(self, name: str, title: str, 
                        description: str = "") -> Dict[str, Any]:
        """
        Create new dashboard
        
        Args:
            name: Dashboard name/ID
            title: Dashboard title
            description: Dashboard description
            
        Returns:
            Dashboard configuration
        """
        dashboard = {
            'name': name,
            'title': title,
            'description': description,
            'panels': [],
            'layout': 'grid',
            'refresh_interval': 30  # seconds
        }
        
        self.dashboards[name] = dashboard
        
        logger.info(f"Dashboard created: {title}")
        
        return dashboard
    
    def add_metric_panel(self, dashboard_name: str, 
                        metric_name: str, 
                        panel_title: str,
                        panel_type: str = "gauge",
                        thresholds: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Add metric panel to dashboard
        
        Args:
            dashboard_name: Dashboard to add to
            metric_name: Metric to display
            panel_title: Panel title
            panel_type: Panel type (gauge, graph, stat, table)
            thresholds: Threshold values for coloring
            
        Returns:
            Panel configuration
        """
        if dashboard_name not in self.dashboards:
            raise ValueError(f"Dashboard not found: {dashboard_name}")
        
        panel = {
            'type': panel_type,
            'title': panel_title,
            'metric': metric_name,
            'thresholds': thresholds or {
                'good': 0.8,
                'warning': 0.6,
                'critical': 0.4
            }
        }
        
        self.dashboards[dashboard_name]['panels'].append(panel)
        
        logger.debug(f"Panel added: {panel_title} to {dashboard_name}")
        
        return panel
    
    def add_time_series_panel(self, dashboard_name: str,
                             metrics: List[str],
                             panel_title: str) -> Dict[str, Any]:
        """
        Add time series graph panel
        
        Args:
            dashboard_name: Dashboard to add to
            metrics: List of metrics to plot
            panel_title: Panel title
            
        Returns:
            Panel configuration
        """
        if dashboard_name not in self.dashboards:
            raise ValueError(f"Dashboard not found: {dashboard_name}")
        
        panel = {
            'type': 'time_series',
            'title': panel_title,
            'metrics': metrics,
            'time_range': '1h'
        }
        
        self.dashboards[dashboard_name]['panels'].append(panel)
        
        return panel
    
    def add_comparison_panel(self, dashboard_name: str,
                            categories: List[str],
                            metric_name: str,
                            panel_title: str) -> Dict[str, Any]:
        """
        Add comparison bar chart panel
        
        Args:
            dashboard_name: Dashboard to add to
            categories: Categories to compare
            metric_name: Metric to compare
            panel_title: Panel title
            
        Returns:
            Panel configuration
        """
        if dashboard_name not in self.dashboards:
            raise ValueError(f"Dashboard not found: {dashboard_name}")
        
        panel = {
            'type': 'bar_chart',
            'title': panel_title,
            'categories': categories,
            'metric': metric_name
        }
        
        self.dashboards[dashboard_name]['panels'].append(panel)
        
        return panel
    
    def add_table_panel(self, dashboard_name: str,
                       columns: List[str],
                       data_source: str,
                       panel_title: str) -> Dict[str, Any]:
        """
        Add data table panel
        
        Args:
            dashboard_name: Dashboard to add to
            columns: Column names
            data_source: Data source identifier
            panel_title: Panel title
            
        Returns:
            Panel configuration
        """
        if dashboard_name not in self.dashboards:
            raise ValueError(f"Dashboard not found: {dashboard_name}")
        
        panel = {
            'type': 'table',
            'title': panel_title,
            'columns': columns,
            'data_source': data_source
        }
        
        self.dashboards[dashboard_name]['panels'].append(panel)
        
        return panel
    
    def create_evaluation_dashboard(self) -> Dict[str, Any]:
        """
        Create standard evaluation dashboard
        
        Returns:
            Dashboard configuration
        """
        dashboard = self.create_dashboard(
            name="evaluation_dashboard",
            title="Agent Evaluation Dashboard",
            description="Comprehensive evaluation metrics"
        )
        
        # Add metric gauges
        self.add_metric_panel(
            "evaluation_dashboard",
            "relevance",
            "Relevance Score",
            panel_type="gauge"
        )
        
        self.add_metric_panel(
            "evaluation_dashboard",
            "coherence",
            "Coherence Score",
            panel_type="gauge"
        )
        
        self.add_metric_panel(
            "evaluation_dashboard",
            "groundedness",
            "Groundedness Score",
            panel_type="gauge"
        )
        
        self.add_metric_panel(
            "evaluation_dashboard",
            "factuality",
            "Factuality Score",
            panel_type="gauge"
        )
        
        # Add time series
        self.add_time_series_panel(
            "evaluation_dashboard",
            ["relevance", "coherence", "groundedness", "factuality"],
            "Metrics Over Time"
        )
        
        # Add comparison chart
        self.add_comparison_panel(
            "evaluation_dashboard",
            ["relevance", "coherence", "groundedness", "factuality"],
            "score",
            "Metric Comparison"
        )
        
        logger.info("Standard evaluation dashboard created")
        
        return dashboard
    
    def create_performance_dashboard(self) -> Dict[str, Any]:
        """
        Create performance monitoring dashboard
        
        Returns:
            Dashboard configuration
        """
        dashboard = self.create_dashboard(
            name="performance_dashboard",
            title="Performance Monitoring",
            description="Agent performance metrics"
        )
        
        # Latency metrics
        self.add_metric_panel(
            "performance_dashboard",
            "avg_latency",
            "Average Latency",
            panel_type="stat"
        )
        
        self.add_metric_panel(
            "performance_dashboard",
            "p95_latency",
            "P95 Latency",
            panel_type="stat"
        )
        
        # Throughput
        self.add_metric_panel(
            "performance_dashboard",
            "throughput",
            "Requests/Second",
            panel_type="stat"
        )
        
        # Error rate
        self.add_metric_panel(
            "performance_dashboard",
            "error_rate",
            "Error Rate",
            panel_type="gauge",
            thresholds={'good': 0.01, 'warning': 0.05, 'critical': 0.1}
        )
        
        # Time series
        self.add_time_series_panel(
            "performance_dashboard",
            ["avg_latency", "throughput", "error_rate"],
            "Performance Trends"
        )
        
        logger.info("Performance dashboard created")
        
        return dashboard
    
    def export_dashboard(self, dashboard_name: str, filepath: str):
        """
        Export dashboard configuration to JSON
        
        Args:
            dashboard_name: Dashboard to export
            filepath: Output file path
        """
        if dashboard_name not in self.dashboards:
            raise ValueError(f"Dashboard not found: {dashboard_name}")
        
        dashboard = self.dashboards[dashboard_name]
        
        with open(filepath, 'w') as f:
            json.dump(dashboard, f, indent=2)
        
        logger.info(f"Dashboard exported: {filepath}")
    
    def get_dashboard(self, dashboard_name: str) -> Optional[Dict[str, Any]]:
        """Get dashboard configuration"""
        return self.dashboards.get(dashboard_name)
    
    def list_dashboards(self) -> List[str]:
        """List all dashboard names"""
        return list(self.dashboards.keys())