"""
Incident Detection Agent

Detects incidents from metrics and alerts
"""

from typing import Dict, Any, List, Optional
import logging
from datetime import datetime
import sys
sys.path.append('../..')

from shared.base_agent import BaseAgent, AgentResponse

logger = logging.getLogger(__name__)


class IncidentDetectionAgent(BaseAgent):
    """
    Agent for detecting incidents from monitoring data
    """
    
    def __init__(self):
        super().__init__(
            name="Incident Detection Agent",
            model="claude-sonnet-4-20250514",
            temperature=0.3
        )
        
        # Detection thresholds
        self.severity_thresholds = {
            "critical": 0.9,
            "high": 0.7,
            "medium": 0.5,
            "low": 0.3
        }
    
    def _get_system_prompt(self) -> str:
        """System prompt for incident detection"""
        return """You are an expert DevOps incident detection system.

Your role:
1. Analyze metrics, logs, and alerts to detect incidents
2. Classify incident severity and type
3. Identify affected services and components
4. Distinguish real incidents from noise
5. Correlate related alerts into single incidents
6. Prioritize incidents by business impact

Detection principles:
- Consider context: time of day, recent changes, normal patterns
- Correlate signals: don't alert on single data points
- Reduce noise: filter false positives
- Act fast: detect incidents quickly
- Be precise: minimize false alarms

Severity levels:
- Critical: Service down, data loss, security breach
- High: Degraded performance, partial outage
- Medium: Minor issues, approaching thresholds
- Low: Informational, potential future issues

Output format:
- Incident detected: yes/no
- Severity: critical/high/medium/low
- Type: outage/degradation/anomaly/threshold
- Affected components
- Evidence and metrics
- Recommended actions"""
    
    async def process(self, input_data: Dict[str, Any]) -> AgentResponse:
        """
        Process monitoring data to detect incidents
        
        Args:
            input_data: {
                "metrics": Dict[str, Any],
                "alerts": List[Dict],
                "logs": Optional[List[str]],
                "timeframe": str,
                "context": Optional[Dict]
            }
        
        Returns:
            AgentResponse with incident detection results
        """
        
        metrics = input_data.get("metrics", {})
        alerts = input_data.get("alerts", [])
        logs = input_data.get("logs", [])
        timeframe = input_data.get("timeframe", "5m")
        context = input_data.get("context", {})
        
        logger.info(f"Analyzing {len(metrics)} metrics, {len(alerts)} alerts")
        
        # Build detection prompt
        detection_prompt = self._build_detection_prompt(
            metrics=metrics,
            alerts=alerts,
            logs=logs,
            timeframe=timeframe,
            context=context
        )
        
        # Analyze for incidents
        messages = [{"role": "user", "content": detection_prompt}]
        analysis = await self._call_llm(messages)
        
        # Parse incidents
        incidents = self._parse_incidents(analysis, metrics, alerts)
        
        return AgentResponse(
            content=analysis,
            agent_name=self.name,
            timestamp=datetime.utcnow(),
            metadata={
                "incidents_detected": len(incidents) > 0,
                "incident_count": len(incidents),
                "incidents": incidents,
                "metrics_analyzed": len(metrics),
                "alerts_analyzed": len(alerts)
            },
            confidence=0.85
        )
    
    def _build_detection_prompt(
        self,
        metrics: Dict[str, Any],
        alerts: List[Dict[str, Any]],
        logs: List[str],
        timeframe: str,
        context: Dict[str, Any]
    ) -> str:
        """Build incident detection prompt"""
        
        prompt_parts = [
            f"Analyze the following monitoring data from the last {timeframe}:",
            ""
        ]
        
        # Add metrics
        if metrics:
            prompt_parts.append("Metrics:")
            for metric_name, value in list(metrics.items())[:20]:
                prompt_parts.append(f"  - {metric_name}: {value}")
            prompt_parts.append("")
        
        # Add alerts
        if alerts:
            prompt_parts.append(f"Active Alerts ({len(alerts)}):")
            for alert in alerts[:10]:
                prompt_parts.append(
                    f"  - {alert.get('name', 'Unknown')}: "
                    f"{alert.get('state', 'unknown')} "
                    f"(severity: {alert.get('severity', 'unknown')})"
                )
            prompt_parts.append("")
        
        # Add recent logs (if provided)
        if logs:
            prompt_parts.append("Recent Error Logs:")
            for log in logs[:5]:
                prompt_parts.append(f"  - {log[:200]}")
            prompt_parts.append("")
        
        # Add context
        if context:
            prompt_parts.append("Context:")
            if "recent_deployments" in context:
                prompt_parts.append("  Recent Deployments:")
                for deploy in context["recent_deployments"][:3]:
                    prompt_parts.append(f"    - {deploy}")
            if "time_of_day" in context:
                prompt_parts.append(f"  Time: {context['time_of_day']}")
            prompt_parts.append("")
        
        prompt_parts.append("""
Analyze this data and determine:
1. Are there any incidents? (yes/no)
2. If yes, for each incident:
   - Title and description
   - Severity (critical/high/medium/low)
   - Type (outage/degradation/anomaly/threshold)
   - Affected services/components
   - Evidence (which metrics/alerts indicate the issue)
   - Recommended immediate actions

Respond in a structured format.""")
        
        return "\n".join(prompt_parts)
    
    def _parse_incidents(
        self,
        analysis: str,
        metrics: Dict[str, Any],
        alerts: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Parse detected incidents from analysis"""
        
        incidents = []
        
        # Simple parsing - in production, use structured output
        analysis_lower = analysis.lower()
        
        if "no incident" in analysis_lower or "no issues detected" in analysis_lower:
            return incidents
        
        # Extract basic incident info
        # This is simplified - in production, use structured LLM output
        
        if any(word in analysis_lower for word in ["critical", "outage", "down"]):
            incidents.append({
                "id": f"inc_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
                "title": "Service Degradation Detected",
                "severity": "high",
                "type": "degradation",
                "affected_components": self._extract_components(metrics, alerts),
                "evidence": alerts[:3],
                "detected_at": datetime.utcnow().isoformat(),
                "status": "open"
            })
        
        return incidents
    
    def _extract_components(
        self,
        metrics: Dict[str, Any],
        alerts: List[Dict[str, Any]]
    ) -> List[str]:
        """Extract affected components from metrics and alerts"""
        
        components = set()
        
        # Extract from alert labels
        for alert in alerts:
            if "labels" in alert:
                if "service" in alert["labels"]:
                    components.add(alert["labels"]["service"])
                if "pod" in alert["labels"]:
                    components.add(alert["labels"]["pod"])
        
        # Extract from metric names
        for metric_name in metrics.keys():
            # Simple extraction - e.g., "api_latency" -> "api"
            parts = metric_name.split("_")
            if parts:
                components.add(parts[0])
        
        return list(components)[:5]