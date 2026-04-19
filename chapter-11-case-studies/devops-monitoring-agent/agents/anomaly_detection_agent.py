"""
Anomaly Detection Agent

Detects anomalies using statistical and ML methods
"""

from typing import Dict, Any, List, Optional
import logging
from datetime import datetime
import sys
sys.path.append('../..')

from shared.base_agent import BaseAgent, AgentResponse
from monitoring.anomaly_detector import AnomalyDetector

logger = logging.getLogger(__name__)


class AnomalyDetectionAgent(BaseAgent):
    """
    Agent for detecting anomalies in metrics and logs
    """
    
    def __init__(self):
        super().__init__(
            name="Anomaly Detection Agent",
            model="claude-sonnet-4-20250514",
            temperature=0.3
        )
        
        self.detector = AnomalyDetector()
    
    def _get_system_prompt(self) -> str:
        """System prompt for anomaly detection"""
        return """You are an expert anomaly detection system for DevOps monitoring.

Your role:
1. Analyze metrics and logs for anomalies
2. Distinguish real anomalies from noise
3. Assess severity and impact of anomalies
4. Correlate related anomalies
5. Provide context and explanations

Detection principles:
- Consider normal patterns and baselines
- Account for time-of-day and seasonal variations
- Look for correlated anomalies across metrics
- Reduce false positives through correlation
- Explain why something is anomalous

Anomaly types:
- Spikes: Sudden increase
- Drops: Sudden decrease
- Trend changes: Direction shifts
- Outliers: Statistical outliers
- Pattern breaks: Deviation from normal patterns

Output format:
- Anomaly detected: yes/no
- Anomaly type and severity
- Affected metrics
- Statistical evidence
- Potential causes
- Recommended investigation steps"""
    
    async def process(self, input_data: Dict[str, Any]) -> AgentResponse:
        """
        Process metrics to detect anomalies
        
        Args:
            input_data: {
                "metrics": Dict[str, float],
                "baseline": Optional[Dict[str, Any]],
                "context": Optional[Dict[str, Any]]
            }
        
        Returns:
            AgentResponse with anomaly detection results
        """
        
        metrics = input_data.get("metrics", {})
        baseline = input_data.get("baseline", {})
        context = input_data.get("context", {})
        
        logger.info(f"Detecting anomalies in {len(metrics)} metrics")
        
        # Use statistical anomaly detector
        statistical_anomalies = self.detector.detect_anomalies(metrics)
        
        # Build analysis prompt
        analysis_prompt = self._build_analysis_prompt(
            metrics=metrics,
            statistical_anomalies=statistical_anomalies,
            baseline=baseline,
            context=context
        )
        
        # Get LLM analysis
        messages = [{"role": "user", "content": analysis_prompt}]
        analysis = await self._call_llm(messages)
        
        # Combine results
        all_anomalies = self._combine_anomalies(
            statistical_anomalies=statistical_anomalies,
            llm_analysis=analysis
        )
        
        return AgentResponse(
            content=analysis,
            agent_name=self.name,
            timestamp=datetime.utcnow(),
            metadata={
                "anomalies_detected": len(all_anomalies) > 0,
                "anomaly_count": len(all_anomalies),
                "anomalies": all_anomalies,
                "statistical_detections": len(statistical_anomalies),
                "metrics_analyzed": len(metrics)
            },
            confidence=0.8
        )
    
    def _build_analysis_prompt(
        self,
        metrics: Dict[str, float],
        statistical_anomalies: List[Dict[str, Any]],
        baseline: Dict[str, Any],
        context: Dict[str, Any]
    ) -> str:
        """Build anomaly analysis prompt"""
        
        prompt_parts = [
            "Analyze the following metrics for anomalies:",
            ""
        ]
        
        # Add current metrics
        if metrics:
            prompt_parts.append("Current Metrics:")
            for metric_name, value in list(metrics.items())[:20]:
                prompt_parts.append(f"  - {metric_name}: {value}")
            prompt_parts.append("")
        
        # Add statistical anomalies
        if statistical_anomalies:
            prompt_parts.append(f"Statistical Anomalies Detected ({len(statistical_anomalies)}):")
            for anomaly in statistical_anomalies:
                prompt_parts.append(
                    f"  - {anomaly['metric']}: {anomaly['type']} "
                    f"(severity: {anomaly['severity']})"
                )
                prompt_parts.append(f"    {anomaly['description']}")
            prompt_parts.append("")
        
        # Add baseline
        if baseline:
            prompt_parts.append("Baseline Statistics:")
            for metric, stats in list(baseline.items())[:10]:
                if isinstance(stats, dict):
                    prompt_parts.append(
                        f"  - {metric}: mean={stats.get('mean', 'N/A')}, "
                        f"stdev={stats.get('stdev', 'N/A')}"
                    )
            prompt_parts.append("")
        
        # Add context
        if context:
            prompt_parts.append("Context:")
            if "time_of_day" in context:
                prompt_parts.append(f"  Time: {context['time_of_day']}")
            if "recent_deployments" in context:
                prompt_parts.append("  Recent deployments: Yes")
            if "load_pattern" in context:
                prompt_parts.append(f"  Load pattern: {context['load_pattern']}")
            prompt_parts.append("")
        
        prompt_parts.append("""
Analyze these metrics and determine:

1. Are the statistical anomalies real issues or false positives?
2. Are there any additional anomalies not caught by statistics?
3. Are any anomalies correlated (indicating a common cause)?
4. What is the severity and potential impact?
5. What are the likely causes?

Provide a comprehensive analysis.""")
        
        return "\n".join(prompt_parts)
    
    def _combine_anomalies(
        self,
        statistical_anomalies: List[Dict[str, Any]],
        llm_analysis: str
    ) -> List[Dict[str, Any]]:
        """Combine statistical and LLM-detected anomalies"""
        
        # Start with statistical anomalies
        all_anomalies = statistical_anomalies.copy()
        
        # Parse LLM analysis for additional anomalies
        # Simplified - in production, use structured output
        
        # Filter false positives mentioned in analysis
        if "false positive" in llm_analysis.lower():
            # LLM identified some statistical anomalies as false positives
            # Keep only high-confidence ones
            all_anomalies = [
                a for a in all_anomalies
                if a.get("severity") in ["high", "critical"]
            ]
        
        return all_anomalies