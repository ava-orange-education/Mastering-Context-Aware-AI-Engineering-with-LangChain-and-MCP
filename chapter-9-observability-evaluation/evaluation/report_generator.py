"""
Generate comprehensive evaluation reports.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ReportGenerator:
    """Generate evaluation reports in various formats"""
    
    def __init__(self):
        """Initialize report generator"""
        self.reports: List[Dict] = []
    
    def generate_text_report(self, evaluation_results: Dict[str, Any]) -> str:
        """
        Generate text-based evaluation report
        
        Args:
            evaluation_results: Evaluation results
            
        Returns:
            Formatted text report
        """
        report_lines = []
        report_lines.append("=" * 70)
        report_lines.append("EVALUATION REPORT")
        report_lines.append("=" * 70)
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        # Summary section
        if 'summary' in evaluation_results:
            summary = evaluation_results['summary']
            report_lines.append("SUMMARY")
            report_lines.append("-" * 70)
            
            if 'average_relevance' in summary:
                report_lines.append(f"Average Relevance:    {summary['average_relevance']:.3f}")
            if 'average_coherence' in summary:
                report_lines.append(f"Average Coherence:    {summary['average_coherence']:.3f}")
            if 'average_groundedness' in summary:
                report_lines.append(f"Average Groundedness: {summary['average_groundedness']:.3f}")
            if 'average_factuality' in summary:
                report_lines.append(f"Average Factuality:   {summary['average_factuality']:.3f}")
            if 'average_overall' in summary:
                report_lines.append(f"Average Overall:      {summary['average_overall']:.3f}")
            
            report_lines.append(f"Total Evaluations:    {summary.get('total_evaluations', 0)}")
            report_lines.append("")
        
        # Individual results
        if 'individual_results' in evaluation_results:
            report_lines.append("INDIVIDUAL RESULTS")
            report_lines.append("-" * 70)
            
            for idx, result in enumerate(evaluation_results['individual_results'][:10], 1):
                report_lines.append(f"\nTest Case {idx}:")
                report_lines.append(f"  Query: {result.get('query', 'N/A')[:60]}...")
                report_lines.append(f"  Overall Score: {result.get('overall_score', 0):.3f}")
                
                if 'relevance' in result:
                    report_lines.append(f"  Relevance: {result['relevance']:.3f}")
                if 'coherence' in result:
                    report_lines.append(f"  Coherence: {result['coherence']:.3f}")
            
            if len(evaluation_results['individual_results']) > 10:
                report_lines.append(f"\n... and {len(evaluation_results['individual_results']) - 10} more results")
        
        report_lines.append("\n" + "=" * 70)
        
        return "\n".join(report_lines)
    
    def generate_json_report(self, evaluation_results: Dict[str, Any], 
                            filepath: Optional[str] = None) -> str:
        """
        Generate JSON evaluation report
        
        Args:
            evaluation_results: Evaluation results
            filepath: Optional file path to save
            
        Returns:
            JSON string
        """
        report = {
            'generated_at': datetime.now().isoformat(),
            'report_type': 'evaluation',
            'results': evaluation_results
        }
        
        json_str = json.dumps(report, indent=2)
        
        if filepath:
            with open(filepath, 'w') as f:
                f.write(json_str)
            logger.info(f"JSON report saved to {filepath}")
        
        return json_str
    
    def generate_html_report(self, evaluation_results: Dict[str, Any]) -> str:
        """
        Generate HTML evaluation report
        
        Args:
            evaluation_results: Evaluation results
            
        Returns:
            HTML string
        """
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Evaluation Report</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 40px;
            background-color: #f5f5f5;
        }}
        .container {{
            background-color: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #333;
            border-bottom: 3px solid #4CAF50;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #555;
            margin-top: 30px;
        }}
        .metric {{
            display: inline-block;
            margin: 10px;
            padding: 15px;
            background-color: #f9f9f9;
            border-radius: 5px;
            min-width: 200px;
        }}
        .metric-label {{
            font-size: 14px;
            color: #666;
        }}
        .metric-value {{
            font-size: 24px;
            font-weight: bold;
            color: #4CAF50;
        }}
        .test-case {{
            margin: 15px 0;
            padding: 15px;
            background-color: #fafafa;
            border-left: 4px solid #2196F3;
        }}
        .score-good {{ color: #4CAF50; }}
        .score-medium {{ color: #FF9800; }}
        .score-poor {{ color: #F44336; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Evaluation Report</h1>
        <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
"""
        
        # Summary section
        if 'summary' in evaluation_results:
            summary = evaluation_results['summary']
            html += "<h2>Summary Metrics</h2>\n<div>\n"
            
            metrics = [
                ('Relevance', summary.get('average_relevance')),
                ('Coherence', summary.get('average_coherence')),
                ('Groundedness', summary.get('average_groundedness')),
                ('Factuality', summary.get('average_factuality')),
                ('Overall', summary.get('average_overall'))
            ]
            
            for label, value in metrics:
                if value is not None:
                    html += f"""
    <div class="metric">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value:.3f}</div>
    </div>
"""
            
            html += "</div>\n"
        
        # Individual results
        if 'individual_results' in evaluation_results:
            html += "<h2>Individual Test Cases</h2>\n"
            
            for idx, result in enumerate(evaluation_results['individual_results'][:10], 1):
                score = result.get('overall_score', 0)
                score_class = 'score-good' if score >= 0.7 else ('score-medium' if score >= 0.4 else 'score-poor')
                
                html += f"""
    <div class="test-case">
        <strong>Test Case {idx}</strong><br>
        <strong>Query:</strong> {result.get('query', 'N/A')[:100]}...<br>
        <strong>Overall Score:</strong> <span class="{score_class}">{score:.3f}</span>
    </div>
"""
        
        html += """
    </div>
</body>
</html>
"""
        
        return html
    
    def generate_markdown_report(self, evaluation_results: Dict[str, Any]) -> str:
        """
        Generate Markdown evaluation report
        
        Args:
            evaluation_results: Evaluation results
            
        Returns:
            Markdown string
        """
        md = "# Evaluation Report\n\n"
        md += f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        # Summary
        if 'summary' in evaluation_results:
            summary = evaluation_results['summary']
            md += "## Summary\n\n"
            md += "| Metric | Score |\n"
            md += "|--------|-------|\n"
            
            if 'average_relevance' in summary:
                md += f"| Relevance | {summary['average_relevance']:.3f} |\n"
            if 'average_coherence' in summary:
                md += f"| Coherence | {summary['average_coherence']:.3f} |\n"
            if 'average_groundedness' in summary:
                md += f"| Groundedness | {summary['average_groundedness']:.3f} |\n"
            if 'average_factuality' in summary:
                md += f"| Factuality | {summary['average_factuality']:.3f} |\n"
            if 'average_overall' in summary:
                md += f"| **Overall** | **{summary['average_overall']:.3f}** |\n"
            
            md += "\n"
        
        # Individual results
        if 'individual_results' in evaluation_results:
            md += "## Individual Results\n\n"
            
            for idx, result in enumerate(evaluation_results['individual_results'][:5], 1):
                md += f"### Test Case {idx}\n\n"
                md += f"**Query:** {result.get('query', 'N/A')}\n\n"
                md += f"**Overall Score:** {result.get('overall_score', 0):.3f}\n\n"
                
                if 'relevance' in result:
                    md += f"- Relevance: {result['relevance']:.3f}\n"
                if 'coherence' in result:
                    md += f"- Coherence: {result['coherence']:.3f}\n"
                
                md += "\n"
        
        return md
    
    def save_report(self, evaluation_results: Dict[str, Any], 
                   filepath: str, format: str = 'json'):
        """
        Save report to file
        
        Args:
            evaluation_results: Evaluation results
            filepath: File path
            format: Report format (json, html, markdown, text)
        """
        if format == 'json':
            content = self.generate_json_report(evaluation_results)
        elif format == 'html':
            content = self.generate_html_report(evaluation_results)
        elif format == 'markdown':
            content = self.generate_markdown_report(evaluation_results)
        else:  # text
            content = self.generate_text_report(evaluation_results)
        
        with open(filepath, 'w') as f:
            f.write(content)
        
        logger.info(f"Report saved to {filepath} ({format} format)")