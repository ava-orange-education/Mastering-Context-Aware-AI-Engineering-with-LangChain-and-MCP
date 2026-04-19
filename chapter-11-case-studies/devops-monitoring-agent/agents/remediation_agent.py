"""
Remediation Agent

Suggests and executes remediation actions for incidents
"""

from typing import Dict, Any, List, Optional
import logging
from datetime import datetime
import sys
sys.path.append('../..')

from shared.base_agent import BaseAgent, AgentResponse

logger = logging.getLogger(__name__)


class RemediationAgent(BaseAgent):
    """
    Agent for incident remediation
    """
    
    def __init__(self):
        super().__init__(
            name="Remediation Agent",
            model="claude-sonnet-4-20250514",
            temperature=0.3
        )
        
        # Safe actions that can be automated
        self.safe_actions = {
            "restart_pod",
            "scale_deployment",
            "clear_cache",
            "reload_config",
            "trigger_gc"
        }
        
        # Actions requiring approval
        self.supervised_actions = {
            "terminate_instance",
            "modify_security_group",
            "change_routing",
            "rollback_deployment",
            "execute_migration"
        }
    
    def _get_system_prompt(self) -> str:
        """System prompt for remediation"""
        return """You are an expert DevOps remediation system.

Your role:
1. Suggest remediation actions for incidents
2. Prioritize actions by safety and effectiveness
3. Provide step-by-step execution plans
4. Assess action risks and impacts
5. Recommend rollback strategies
6. Learn from action outcomes

Remediation principles:
- Safety First: Never make things worse
- Minimize Impact: Limit blast radius
- Quick Wins: Start with low-risk, high-impact actions
- Reversibility: Prefer reversible actions
- Progressive: Escalate from safe to riskier actions
- Documentation: Explain why each action helps

Action safety levels:
- Safe (auto-executable): restart pod, scale up, clear cache
- Supervised (needs approval): rollback, routing changes, instance termination
- High-risk (manual only): database changes, security modifications

Output format:
- Recommended actions (prioritized list)
- For each action:
  - What it does
  - Why it helps
  - Safety level
  - Expected impact
  - Rollback plan
  - Execution steps"""
    
    async def process(self, input_data: Dict[str, Any]) -> AgentResponse:
        """
        Generate remediation plan
        
        Args:
            input_data: {
                "incident": Dict[str, Any],
                "root_cause": Optional[str],
                "system_state": Dict[str, Any],
                "allow_automatic": bool,
                "constraints": Optional[Dict]
            }
        
        Returns:
            AgentResponse with remediation plan
        """
        
        incident = input_data.get("incident", {})
        root_cause = input_data.get("root_cause")
        system_state = input_data.get("system_state", {})
        allow_automatic = input_data.get("allow_automatic", False)
        constraints = input_data.get("constraints", {})
        
        logger.info(f"Generating remediation plan for: {incident.get('title', 'Unknown')}")
        
        # Build remediation prompt
        remediation_prompt = self._build_remediation_prompt(
            incident=incident,
            root_cause=root_cause,
            system_state=system_state,
            allow_automatic=allow_automatic,
            constraints=constraints
        )
        
        # Generate plan
        messages = [{"role": "user", "content": remediation_prompt}]
        plan = await self._call_llm(messages)
        
        # Extract and validate actions
        actions = self._extract_actions(plan, allow_automatic)
        
        return AgentResponse(
            content=plan,
            agent_name=self.name,
            timestamp=datetime.utcnow(),
            metadata={
                "actions": actions,
                "safe_actions": [a for a in actions if a["safe_to_automate"]],
                "supervised_actions": [a for a in actions if not a["safe_to_automate"]],
                "total_actions": len(actions)
            },
            confidence=0.8
        )
    
    def _build_remediation_prompt(
        self,
        incident: Dict[str, Any],
        root_cause: Optional[str],
        system_state: Dict[str, Any],
        allow_automatic: bool,
        constraints: Dict[str, Any]
    ) -> str:
        """Build remediation planning prompt"""
        
        prompt_parts = [
            "Generate a remediation plan for the following incident:",
            "",
            f"Incident: {incident.get('title', 'Unknown')}",
            f"Severity: {incident.get('severity', 'unknown')}",
            f"Type: {incident.get('type', 'unknown')}",
        ]
        
        if root_cause:
            prompt_parts.append(f"Root Cause: {root_cause}")
        
        prompt_parts.append("")
        
        # Add affected components
        if "affected_components" in incident:
            prompt_parts.append("Affected Components:")
            for component in incident["affected_components"]:
                prompt_parts.append(f"  - {component}")
            prompt_parts.append("")
        
        # Add system state
        if system_state:
            prompt_parts.append("Current System State:")
            for key, value in list(system_state.items())[:10]:
                prompt_parts.append(f"  - {key}: {value}")
            prompt_parts.append("")
        
        # Add constraints
        if constraints:
            prompt_parts.append("Constraints:")
            for key, value in constraints.items():
                prompt_parts.append(f"  - {key}: {value}")
            prompt_parts.append("")
        
        prompt_parts.append(f"""
Automation Level: {"Automatic actions allowed" if allow_automatic else "Supervised only"}

Safe Actions (can be automated):
- restart_pod: Restart unhealthy pods
- scale_deployment: Scale replicas up/down
- clear_cache: Clear application cache
- reload_config: Reload configuration
- trigger_gc: Trigger garbage collection

Supervised Actions (need approval):
- rollback_deployment: Rollback to previous version
- terminate_instance: Terminate EC2/VM instance
- modify_security_group: Change firewall rules
- change_routing: Modify traffic routing

Generate a prioritized remediation plan:

1. Immediate Actions (to stop the bleeding):
   - What to do first
   - Expected impact
   - Safety level

2. Short-term Fixes (to restore service):
   - Actions to resolve the incident
   - Step-by-step execution
   - Rollback procedures

3. Long-term Improvements (to prevent recurrence):
   - Architectural changes
   - Monitoring improvements
   - Process updates

For each action, specify:
- Action name
- Description
- Why it helps
- Safety level (safe/supervised/high-risk)
- Expected impact
- Execution steps
- Rollback plan""")
        
        return "\n".join(prompt_parts)
    
    def _extract_actions(
        self,
        plan: str,
        allow_automatic: bool
    ) -> List[Dict[str, Any]]:
        """Extract structured actions from plan"""
        
        actions = []
        
        # Simplified extraction - in production, use structured output
        # This is a placeholder implementation
        
        # Look for common action patterns
        action_keywords = {
            "restart": "restart_pod",
            "scale": "scale_deployment",
            "rollback": "rollback_deployment",
            "clear cache": "clear_cache"
        }
        
        for keyword, action_name in action_keywords.items():
            if keyword in plan.lower():
                is_safe = action_name in self.safe_actions
                
                actions.append({
                    "action": action_name,
                    "description": f"Execute {action_name.replace('_', ' ')}",
                    "safe_to_automate": is_safe and allow_automatic,
                    "requires_approval": not is_safe,
                    "impact": "medium",
                    "execution_steps": [
                        f"Validate {action_name} is appropriate",
                        f"Execute {action_name}",
                        "Monitor results"
                    ],
                    "rollback_plan": f"Undo {action_name} if issues occur"
                })
        
        return actions[:5]  # Limit to top 5 actions
    
    def validate_action(
        self,
        action: Dict[str, Any],
        system_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Validate action before execution
        
        Args:
            action: Action to validate
            system_state: Current system state
        
        Returns:
            Validation result
        """
        
        validation = {
            "valid": True,
            "warnings": [],
            "blockers": []
        }
        
        action_name = action.get("action")
        
        # Check if action is known
        if action_name not in (self.safe_actions | self.supervised_actions):
            validation["valid"] = False
            validation["blockers"].append(f"Unknown action: {action_name}")
            return validation
        
        # Action-specific validation
        if action_name == "scale_deployment":
            # Check if we're already at max scale
            current_replicas = system_state.get("replicas", 1)
            max_replicas = system_state.get("max_replicas", 10)
            
            if current_replicas >= max_replicas:
                validation["warnings"].append("Already at maximum replicas")
        
        elif action_name == "restart_pod":
            # Check if pod is in valid state for restart
            pod_state = system_state.get("pod_state", "Running")
            if pod_state == "Terminating":
                validation["blockers"].append("Pod already terminating")
                validation["valid"] = False
        
        return validation