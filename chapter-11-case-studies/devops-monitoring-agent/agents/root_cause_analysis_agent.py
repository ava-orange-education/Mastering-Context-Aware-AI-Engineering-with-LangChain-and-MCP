"""
Root Cause Analysis Agent

Performs comprehensive root cause analysis
"""

from typing import Dict, Any, List, Optional
import logging
from datetime import datetime
import sys
sys.path.append('../..')

from shared.base_agent import BaseAgent, AgentResponse

logger = logging.getLogger(__name__)


class RootCauseAnalysisAgent(BaseAgent):
    """
    Agent for comprehensive root cause analysis
    """
    
    def __init__(self):
        super().__init__(
            name="Root Cause Analysis Agent",
            model="claude-sonnet-4-20250514",
            temperature=0.2
        )
    
    def _get_system_prompt(self) -> str:
        """System prompt for root cause analysis"""
        return """You are an expert DevOps engineer performing root cause analysis.

Your role:
1. Systematically investigate incidents
2. Identify the primary root cause
3. Find contributing factors
4. Provide evidence-based conclusions
5. Recommend preventive measures

Analysis methodology:
- 5 Whys: Ask why repeatedly to find root cause
- Timeline Analysis: Reconstruct event sequence
- Correlation Analysis: Find related changes
- Dependency Tracing: Follow system dependencies
- Pattern Matching: Compare with historical incidents

Root cause criteria:
- Specific and actionable
- Supported by evidence
- Explains all symptoms
- When fixed, prevents recurrence

Output structure:
1. Executive Summary
2. Timeline of Events
3. Primary Root Cause
4. Contributing Factors
5. Evidence Chain
6. Immediate Fixes
7. Long-term Prevention
8. Lessons Learned"""
    
    async def process(self, input_data: Dict[str, Any]) -> AgentResponse:
        """
        Perform root cause analysis
        
        Args:
            input_data: {
                "incident": Dict[str, Any],
                "metrics": Dict[str, Any],
                "logs": List[str],
                "recent_changes": List[Dict],
                "system_state": Dict[str, Any]
            }
        
        Returns:
            AgentResponse with RCA results
        """
        
        incident = input_data.get("incident", {})
        metrics = input_data.get("metrics", {})
        logs = input_data.get("logs", [])
        recent_changes = input_data.get("recent_changes", [])
        system_state = input_data.get("system_state", {})
        
        logger.info(f"Performing RCA for: {incident.get('title', 'Unknown')}")
        
        # Build comprehensive RCA prompt
        rca_prompt = self._build_rca_prompt(
            incident=incident,
            metrics=metrics,
            logs=logs,
            recent_changes=recent_changes,
            system_state=system_state
        )
        
        # Perform analysis
        messages = [{"role": "user", "content": rca_prompt}]
        analysis = await self._call_llm(messages)
        
        # Extract structured RCA
        rca_details = self._extract_rca_details(analysis)
        
        return AgentResponse(
            content=analysis,
            agent_name=self.name,
            timestamp=datetime.utcnow(),
            metadata={
                "root_cause": rca_details.get("root_cause"),
                "contributing_factors": rca_details.get("contributing_factors", []),
                "evidence": rca_details.get("evidence", []),
                "timeline": rca_details.get("timeline", []),
                "prevention_measures": rca_details.get("prevention_measures", []),
                "confidence": rca_details.get("confidence", 0.7)
            },
            confidence=rca_details.get("confidence", 0.7)
        )
    
    def _build_rca_prompt(
        self,
        incident: Dict[str, Any],
        metrics: Dict[str, Any],
        logs: List[str],
        recent_changes: List[Dict[str, Any]],
        system_state: Dict[str, Any]
    ) -> str:
        """Build comprehensive RCA prompt"""
        
        prompt_parts = [
            "="*70,
            "ROOT CAUSE ANALYSIS",
            "="*70,
            "",
            f"Incident: {incident.get('title', 'Unknown')}",
            f"Severity: {incident.get('severity', 'unknown')}",
            f"Started: {incident.get('started_at', 'unknown')}",
            f"Duration: {incident.get('duration', 'unknown')}",
            "",
            "SYMPTOMS:",
            f"{incident.get('description', 'No description')}",
            ""
        ]
        
        # Affected services
        if "affected_components" in incident:
            prompt_parts.append("AFFECTED COMPONENTS:")
            for component in incident["affected_components"]:
                prompt_parts.append(f"  • {component}")
            prompt_parts.append("")
        
        # Metrics at incident time
        if metrics:
            prompt_parts.append("METRICS DURING INCIDENT:")
            for metric_name, value in list(metrics.items())[:15]:
                prompt_parts.append(f"  • {metric_name}: {value}")
            prompt_parts.append("")
        
        # Error logs
        if logs:
            prompt_parts.append("ERROR LOGS:")
            for log in logs[:10]:
                prompt_parts.append(f"  • {log[:200]}")
            prompt_parts.append("")
        
        # Recent changes
        if recent_changes:
            prompt_parts.append("RECENT CHANGES (Last 24h):")
            for change in recent_changes[:5]:
                change_type = change.get("type", "unknown")
                description = change.get("description", "")
                timestamp = change.get("timestamp", "")
                prompt_parts.append(f"  • [{change_type}] {description} at {timestamp}")
            prompt_parts.append("")
        
        # System state
        if system_state:
            prompt_parts.append("SYSTEM STATE:")
            for key, value in list(system_state.items())[:10]:
                prompt_parts.append(f"  • {key}: {value}")
            prompt_parts.append("")
        
        prompt_parts.extend([
            "="*70,
            "PERFORM ROOT CAUSE ANALYSIS:",
            "="*70,
            "",
            "1. TIMELINE RECONSTRUCTION:",
            "   - What happened first?",
            "   - What was the sequence of events?",
            "   - When did each symptom appear?",
            "",
            "2. 5 WHYS ANALYSIS:",
            "   - Why did this incident occur?",
            "   - Why did that happen? (repeat 5 times)",
            "",
            "3. EVIDENCE ANALYSIS:",
            "   - What metrics show the problem?",
            "   - What do logs reveal?",
            "   - What changes correlate with the incident?",
            "",
            "4. ROOT CAUSE IDENTIFICATION:",
            "   - What is the PRIMARY root cause?",
            "   - What are CONTRIBUTING factors?",
            "   - What EVIDENCE supports this conclusion?",
            "",
            "5. PREVENTION:",
            "   - How can this be prevented in the future?",
            "   - What monitoring should be added?",
            "   - What processes should change?",
            "",
            "Provide a comprehensive, evidence-based analysis."
        ])
        
        return "\n".join(prompt_parts)
    
    def _extract_rca_details(self, analysis: str) -> Dict[str, Any]:
        """Extract structured RCA details"""
        
        details = {
            "root_cause": "Analysis in progress",
            "contributing_factors": [],
            "evidence": [],
            "timeline": [],
            "prevention_measures": [],
            "confidence": 0.7
        }
        
        # Extract root cause
        for line in analysis.split('\n'):
            if "root cause:" in line.lower() or "primary cause:" in line.lower():
                details["root_cause"] = line.split(":", 1)[1].strip()
                break
        
        # Extract contributing factors
        in_factors = False
        for line in analysis.split('\n'):
            if "contributing factor" in line.lower():
                in_factors = True
                continue
            if in_factors and line.strip().startswith(("•", "-", "—")):
                factor = line.strip()[1:].strip()
                if factor:
                    details["contributing_factors"].append(factor)
            elif in_factors and not line.strip():
                break
        
        # Extract prevention measures
        in_prevention = False
        for line in analysis.split('\n'):
            if "prevention" in line.lower() or "preventive" in line.lower():
                in_prevention = True
                continue
            if in_prevention and line.strip().startswith(("•", "-", "—")):
                measure = line.strip()[1:].strip()
                if measure:
                    details["prevention_measures"].append(measure)
            elif in_prevention and not line.strip():
                break
        
        return details