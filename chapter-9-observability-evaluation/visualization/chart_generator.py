"""
Generate charts and visualizations for evaluation results.
"""

from typing import Dict, Any, List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChartGenerator:
    """Generate charts for evaluation data"""
    
    def __init__(self):
        """Initialize chart generator"""
        self.charts = []
    
    def generate_line_chart(self, data: List[Dict[str, Any]], 
                           x_key: str, y_keys: List[str],
                           title: str = "Line Chart") -> Dict[str, Any]:
        """
        Generate line chart configuration
        
        Args:
            data: List of data points
            x_key: Key for x-axis
            y_keys: Keys for y-axis values
            title: Chart title
            
        Returns:
            Chart configuration
        """
        chart = {
            'type': 'line',
            'title': title,
            'data': {
                'x': [point[x_key] for point in data],
                'series': []
            }
        }
        
        for key in y_keys:
            chart['data']['series'].append({
                'name': key,
                'values': [point.get(key, 0) for point in data]
            })
        
        self.charts.append(chart)
        
        return chart
    
    def generate_bar_chart(self, categories: List[str], 
                          values: List[float],
                          title: str = "Bar Chart") -> Dict[str, Any]:
        """
        Generate bar chart configuration
        
        Args:
            categories: Category names
            values: Values for each category
            title: Chart title
            
        Returns:
            Chart configuration
        """
        chart = {
            'type': 'bar',
            'title': title,
            'data': {
                'categories': categories,
                'values': values
            }
        }
        
        self.charts.append(chart)
        
        return chart
    
    def generate_scatter_plot(self, x_values: List[float],
                             y_values: List[float],
                             labels: Optional[List[str]] = None,
                             title: str = "Scatter Plot") -> Dict[str, Any]:
        """
        Generate scatter plot configuration
        
        Args:
            x_values: X-axis values
            y_values: Y-axis values
            labels: Point labels
            title: Chart title
            
        Returns:
            Chart configuration
        """
        chart = {
            'type': 'scatter',
            'title': title,
            'data': {
                'points': [
                    {'x': x, 'y': y, 'label': label}
                    for x, y, label in zip(
                        x_values, 
                        y_values, 
                        labels or [''] * len(x_values)
                    )
                ]
            }
        }
        
        self.charts.append(chart)
        
        return chart
    
    def generate_heatmap(self, data: List[List[float]],
                        x_labels: List[str],
                        y_labels: List[str],
                        title: str = "Heatmap") -> Dict[str, Any]:
        """
        Generate heatmap configuration
        
        Args:
            data: 2D array of values
            x_labels: X-axis labels
            y_labels: Y-axis labels
            title: Chart title
            
        Returns:
            Chart configuration
        """
        chart = {
            'type': 'heatmap',
            'title': title,
            'data': {
                'values': data,
                'x_labels': x_labels,
                'y_labels': y_labels
            }
        }
        
        self.charts.append(chart)
        
        return chart
    
    def generate_metrics_comparison(self, metrics: Dict[str, float],
                                   title: str = "Metrics Comparison") -> Dict[str, Any]:
        """
        Generate comparison chart for multiple metrics
        
        Args:
            metrics: Dictionary of metric names and values
            title: Chart title
            
        Returns:
            Chart configuration
        """
        return self.generate_bar_chart(
            categories=list(metrics.keys()),
            values=list(metrics.values()),
            title=title
        )
    
    def generate_trend_chart(self, timestamps: List[str],
                            metric_values: Dict[str, List[float]],
                            title: str = "Trend Analysis") -> Dict[str, Any]:
        """
        Generate trend chart with multiple metrics over time
        
        Args:
            timestamps: List of timestamps
            metric_values: Dict of metric names to value lists
            title: Chart title
            
        Returns:
            Chart configuration
        """
        data = [{'timestamp': ts} for ts in timestamps]
        
        for metric_name, values in metric_values.items():
            for i, value in enumerate(values):
                if i < len(data):
                    data[i][metric_name] = value
        
        return self.generate_line_chart(
            data=data,
            x_key='timestamp',
            y_keys=list(metric_values.keys()),
            title=title
        )
    
    def export_charts_html(self, filepath: str):
        """
        Export all charts as HTML
        
        Args:
            filepath: Output file path
        """
        html = """
<!DOCTYPE html>
<html>
<head>
    <title>Evaluation Charts</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .chart { margin: 30px 0; }
        h1 { color: #333; }
    </style>
</head>
<body>
    <h1>Evaluation Charts</h1>
"""
        
        for idx, chart in enumerate(self.charts):
            html += f'<div class="chart" id="chart{idx}"></div>\n'
            html += '<script>\n'
            html += self._generate_plotly_script(chart, f'chart{idx}')
            html += '</script>\n'
        
        html += """
</body>
</html>
"""
        
        with open(filepath, 'w') as f:
            f.write(html)
        
        logger.info(f"Charts exported to HTML: {filepath}")
    
    def _generate_plotly_script(self, chart: Dict, div_id: str) -> str:
        """Generate Plotly JavaScript for chart"""
        
        if chart['type'] == 'line':
            traces = []
            for series in chart['data']['series']:
                traces.append(f"""{{
                    x: {chart['data']['x']},
                    y: {series['values']},
                    name: '{series['name']}',
                    type: 'scatter',
                    mode: 'lines+markers'
                }}""")
            
            data = f"[{','.join(traces)}]"
            
        elif chart['type'] == 'bar':
            data = f"""[{{
                x: {chart['data']['categories']},
                y: {chart['data']['values']},
                type: 'bar'
            }}]"""
        
        elif chart['type'] == 'scatter':
            x_vals = [p['x'] for p in chart['data']['points']]
            y_vals = [p['y'] for p in chart['data']['points']]
            labels = [p['label'] for p in chart['data']['points']]
            
            data = f"""[{{
                x: {x_vals},
                y: {y_vals},
                text: {labels},
                mode: 'markers',
                type: 'scatter',
                marker: {{ size: 10 }}
            }}]"""
        
        else:
            data = "[]"
        
        layout = f"""{{
            title: '{chart['title']}',
            xaxis: {{ title: 'X Axis' }},
            yaxis: {{ title: 'Y Axis' }}
        }}"""
        
        return f"Plotly.newPlot('{div_id}', {data}, {layout});\n"