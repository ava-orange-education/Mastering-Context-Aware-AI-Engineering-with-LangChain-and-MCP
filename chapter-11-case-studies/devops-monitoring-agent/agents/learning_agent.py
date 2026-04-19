"""
Learning Agent

Learns from incident resolutions and improves over time
"""

from typing import Dict, Any, List, Optional
import logging
from datetime import datetime
from collections import defaultdict
import sys
sys.path.append('../..')

from shared.base_agent import BaseAgent, AgentResponse

logger = logging.getLogger(__name__)


class LearningAgent(BaseAgent):
    """
    Agent that learns from incident history
    """
    
    def __init__(self):
        super().__init__(
            name="Learning Agent",
            model="claude-sonnet-4-20250514",
            temperature=0.3
        )
        
        # Learning data
        self.incident_patterns = defaultdict(list)
        self.action_effectiveness = defaultdict(list)
        self.runbooks = {}
    
    def _get_system_prompt(self) -> str:
        """System prompt for learning"""
        return """You are an AI system that learns from DevOps incidents and improves over time.

Your role:
1. Analyze resolved incidents for patterns
2. Evaluate effectiveness of remediation actions
3. Generate runbooks from successful resolutions
4. Identify recurring issues
5. Recommend preventive measures
6. Update knowledge base

Learning principles:
- Focus on successful resolutions
- Identify root cause patterns
- Track action effectiveness
- Recognize similar incidents
- Generalize solutions
- Continuous improvement

Output format:
- Patterns identified
- Effective actions
- Generated runbooks
- Recommendations for prevention
- Knowledge base updates"""
    
    async def process(self, input_data: Dict[str, Any]) -> AgentResponse:
        """
        Learn from incident data
        
        Args:
            input_data: {
                "incident": Dict[str, Any],
                "resolution": Dict[str, Any],
                "actions_taken": List[Dict],
                "outcome": str
            }
        
        Returns:
            AgentResponse with learning insights
        """
        
        incident = input_data.get("incident", {})
        resolution = input_data.get("resolution", {})
        actions_taken = input_data.get("actions_taken", [])
        outcome = input_data.get("outcome", "unknown")
        
        logger.info(f"Learning from incident: {incident.get('title', 'Unknown')}")
        
        # Record incident pattern
        self._record_incident_pattern(incident, resolution)
        
        # Record action effectiveness
        self._record_action_effectiveness(actions_taken, outcome)
        
        # Generate insights
        insights = await self._generate_insights(
            incident=incident,
            resolution=resolution,
            actions_taken=actions_taken
        )
        
        # Check if runbook should be generated
        runbook = None
        if self._should_generate_runbook(incident):
            runbook = await self._generate_runbook(incident, resolution, actions_taken)
        
        return AgentResponse(
            content=insights,
            agent_name=self.name,
            timestamp=datetime.utcnow(),
            metadata={
                "patterns_recorded": len(self.incident_patterns),
                "runbook_generated": runbook is not None,
                "runbook": runbook,
                "learning_applied": True
            },
            confidence=0.85
        )
    
    def _record_incident_pattern(
        self,
        incident: Dict[str, Any],
        resolution: Dict[str, Any]
    ) -> None:
        """Record incident pattern for learning"""
        
        incident_type = incident.get("type", "unknown")
        
        pattern = {
            "symptoms": incident.get("symptoms", []),
            "affected_components": incident.get("affected_components", []),
            "root_cause": resolution.get("root_cause"),
            "resolution_method": resolution.get("method"),
            "resolution_time": resolution.get("time_to_resolve"),
            "timestamp": datetime.utcnow()
        }
        
        self.incident_patterns[incident_type].append(pattern)
        
        logger.info(f"Recorded pattern for {incident_type}")
    
    def _record_action_effectiveness(
        self,
        actions: List[Dict[str, Any]],
        outcome: str
    ) -> None:
        """Record effectiveness of actions taken"""
        
        success = outcome == "resolved"
        
        for action in actions:
            action_name = action.get("action")
            
            self.action_effectiveness[action_name].append({
                "success": success,
                "execution_time": action.get("execution_time"),
                "timestamp": datetime.utcnow()
            })
    
    async def _generate_insights(
        self,
        incident: Dict[str, Any],
        resolution: Dict[str, Any],
        actions_taken: List[Dict[str, Any]]
    ) -> str:
        """Generate learning insights"""
        
        # Build insights prompt
        prompt_parts = [
            "Analyze this resolved incident and extract learnings:",
            "",
            f"Incident: {incident.get('title', 'Unknown')}",
            f"Type: {incident.get('type', 'unknown')}",
            f"Root Cause: {resolution.get('root_cause', 'Unknown')}",
            f"Resolution Time: {resolution.get('time_to_resolve', 'Unknown')}",
            ""
        ]
        
        if actions_taken:
            prompt_parts.append("Actions Taken:")
            for action in actions_taken:
                prompt_parts.append(f"  - {action.get('action')}")
            prompt_parts.append("")
        
        # Add historical context
        incident_type = incident.get("type", "unknown")
        if incident_type in self.incident_patterns:
            similar_count = len(self.incident_patterns[incident_type])
            prompt_parts.append(f"Similar incidents seen: {similar_count}")
            prompt_parts.append("")
        
        prompt_parts.append("""
Extract key learnings:
1. What patterns emerged?
2. What worked well?
3. What could be improved?
4. Are there preventive measures?
5. Should this be documented as a runbook?

Provide actionable insights.""")
        
        prompt = "\n".join(prompt_parts)
        
        messages = [{"role": "user", "content": prompt}]
        insights = await self._call_llm(messages)
        
        return insights
    
    def _should_generate_runbook(self, incident: Dict[str, Any]) -> bool:
        """Determine if runbook should be generated"""
        
        incident_type = incident.get("type", "unknown")
        
        # Generate runbook if we've seen this type 3+ times
        if incident_type in self.incident_patterns:
            count = len(self.incident_patterns[incident_type])
            return count >= 3
        
        return False
    
    async def _generate_runbook(
        self,
        incident: Dict[str, Any],
        resolution: Dict[str, Any],
        actions_taken: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate runbook from incident pattern"""
        
        incident_type = incident.get("type", "unknown")
        patterns = self.incident_patterns[incident_type]
        
        # Build runbook generation prompt
        prompt_parts = [
            f"Generate a runbook for {incident_type} incidents based on these patterns:",
            ""
        ]
        
        # Common symptoms
        all_symptoms = []
        for pattern in patterns:
            all_symptoms.extend(pattern.get("symptoms", []))
        
        unique_symptoms = list(set(all_symptoms))
        if unique_symptoms:
            prompt_parts.append("Common Symptoms:")
            for symptom in unique_symptoms[:5]:
                prompt_parts.append(f"  - {symptom}")
            prompt_parts.append("")
        
        # Common root causes
        root_causes = [p.get("root_cause") for p in patterns if p.get("root_cause")]
        if root_causes:
            prompt_parts.append("Common Root Causes:")
            for cause in list(set(root_causes))[:3]:
                prompt_parts.append(f"  - {cause}")
            prompt_parts.append("")
        
        # Successful actions
        prompt_parts.append("Actions Taken in Past Incidents:")
        for action in actions_taken:
            prompt_parts.append(f"  - {action.get('action')}")
        prompt_parts.append("")
        
        prompt_parts.append("""
Generate a comprehensive runbook with:
1. Title and purpose
2. Trigger conditions
3. Diagnosis steps
4. Remediation steps
5. Prevention measures

Format as a structured runbook.""")
        
        prompt = "\n".join(prompt_parts)
        
        messages = [{"role": "user", "content": prompt}]
        runbook_text = await self._call_llm(messages)
        
        runbook = {
            "id": f"runbook_{incident_type}",
            "title": f"Runbook: {incident_type}",
            "incident_type": incident_type,
            "content": runbook_text,
            "generated_from": len(patterns),
            "created_at": datetime.utcnow().isoformat()
        }
        
        # Store runbook
        self.runbooks[runbook["id"]] = runbook
        
        logger.info(f"Generated runbook: {runbook['id']}")
        
        return runbook
    
    def get_action_effectiveness_stats(self) -> Dict[str, Any]:
        """Get statistics on action effectiveness"""
        
        stats = {}
        
        for action_name, records in self.action_effectiveness.items():
            if not records:
                continue
            
            total = len(records)
            successful = sum(1 for r in records if r.get("success"))
            
            stats[action_name] = {
                "total_uses": total,
                "successful": successful,
                "success_rate": successful / total if total > 0 else 0,
                "average_time": sum(
                    r.get("execution_time", 0) for r in records
                ) / total if total > 0 else 0
            }
        
        return stats
    
    def get_incident_patterns(
        self,
        incident_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get learned incident patterns"""
        
        if incident_type:
            return {
                "incident_type": incident_type,
                "occurrences": len(self.incident_patterns.get(incident_type, [])),
                "patterns": self.incident_patterns.get(incident_type, [])
            }
        else:
            return {
                "total_types": len(self.incident_patterns),
                "patterns_by_type": {
                    incident_type: len(patterns)
                    for incident_type, patterns in self.incident_patterns.items()
                }
            }