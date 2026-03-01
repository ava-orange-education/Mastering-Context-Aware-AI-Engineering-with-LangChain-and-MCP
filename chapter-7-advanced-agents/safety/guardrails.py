"""
Safety guardrails and constraints for agent behavior.
"""

from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class SafetyRule:
    """Defines a safety constraint"""
    name: str
    description: str
    check_function: Callable[[Any], bool]
    severity: str = "warning"  # warning, error, critical
    violation_message: str = ""


class SafetyGuardrails:
    """Enforces safety constraints on agent actions"""
    
    def __init__(self):
        self.rules: List[SafetyRule] = []
        self.violations: List[Dict[str, Any]] = []
        self.blocked_actions: List[str] = []
    
    def add_rule(self, rule: SafetyRule):
        """Add safety rule"""
        self.rules.append(rule)
        logger.info(f"Added safety rule: {rule.name}")
    
    def add_blocked_action(self, action_name: str):
        """Block specific action"""
        self.blocked_actions.append(action_name)
        logger.info(f"Blocked action: {action_name}")
    
    def check_action(self, action_type: str, action_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check if action is safe
        
        Args:
            action_type: Type of action
            action_data: Action parameters
            
        Returns:
            Safety check result
        """
        # Check if action is blocked
        if action_type in self.blocked_actions:
            violation = {
                'rule': 'blocked_action',
                'action': action_type,
                'severity': 'critical',
                'message': f"Action {action_type} is blocked"
            }
            self.violations.append(violation)
            
            return {
                'allowed': False,
                'violations': [violation]
            }
        
        violations = []
        
        # Check all rules
        for rule in self.rules:
            try:
                if not rule.check_function(action_data):
                    violation = {
                        'rule': rule.name,
                        'action': action_type,
                        'severity': rule.severity,
                        'message': rule.violation_message or f"Violated rule: {rule.name}"
                    }
                    violations.append(violation)
                    self.violations.append(violation)
            except Exception as e:
                logger.error(f"Error checking rule {rule.name}: {e}")
        
        # Determine if action is allowed
        critical_violations = [v for v in violations if v['severity'] == 'critical']
        error_violations = [v for v in violations if v['severity'] == 'error']
        
        allowed = len(critical_violations) == 0 and len(error_violations) == 0
        
        return {
            'allowed': allowed,
            'violations': violations,
            'warnings': [v for v in violations if v['severity'] == 'warning']
        }
    
    def get_violation_summary(self) -> Dict[str, Any]:
        """Get summary of violations"""
        by_severity = {
            'critical': [v for v in self.violations if v['severity'] == 'critical'],
            'error': [v for v in self.violations if v['severity'] == 'error'],
            'warning': [v for v in self.violations if v['severity'] == 'warning']
        }
        
        return {
            'total_violations': len(self.violations),
            'by_severity': {k: len(v) for k, v in by_severity.items()},
            'recent_violations': self.violations[-10:]
        }


class ContentSafetyFilter:
    """Filters unsafe content in agent inputs/outputs"""
    
    def __init__(self):
        self.blocked_patterns = [
            # Add patterns for harmful content
            r'(?i)\b(kill|harm|attack|destroy)\b',
            r'(?i)\b(password|api[_\s]?key|secret|token)\b'
        ]
    
    def check_content(self, text: str) -> Dict[str, Any]:
        """
        Check content for safety issues
        
        Args:
            text: Content to check
            
        Returns:
            Safety check result
        """
        import re
        
        issues = []
        
        for pattern in self.blocked_patterns:
            matches = re.findall(pattern, text)
            if matches:
                issues.append({
                    'pattern': pattern,
                    'matches': matches,
                    'severity': 'warning'
                })
        
        return {
            'safe': len(issues) == 0,
            'issues': issues
        }
    
    def sanitize_content(self, text: str) -> str:
        """Remove or mask unsafe content"""
        import re
        
        sanitized = text
        
        # Mask sensitive patterns
        for pattern in self.blocked_patterns:
            sanitized = re.sub(pattern, '[REDACTED]', sanitized, flags=re.IGNORECASE)
        
        return sanitized


class ActionValidator:
    """Validates agent actions before execution"""
    
    def __init__(self, allowed_tools: Optional[List[str]] = None):
        """
        Initialize action validator
        
        Args:
            allowed_tools: List of allowed tool names (None = all allowed)
        """
        self.allowed_tools = allowed_tools
        self.max_iterations = 50
        self.require_approval = False
        self.approval_callback: Optional[Callable] = None
    
    def set_approval_required(self, callback: Callable):
        """
        Set approval requirement for actions
        
        Args:
            callback: Function that approves/denies actions
        """
        self.require_approval = True
        self.approval_callback = callback
    
    def validate_action(self, action_type: str, action_data: Dict[str, Any], 
                       iteration: int) -> Dict[str, Any]:
        """
        Validate action
        
        Args:
            action_type: Action type
            action_data: Action parameters
            iteration: Current iteration number
            
        Returns:
            Validation result
        """
        issues = []
        
        # Check iteration limit
        if iteration >= self.max_iterations:
            issues.append({
                'type': 'max_iterations',
                'message': f"Exceeded maximum iterations: {self.max_iterations}"
            })
        
        # Check if tool is allowed
        if self.allowed_tools and action_type not in self.allowed_tools:
            issues.append({
                'type': 'unauthorized_tool',
                'message': f"Tool {action_type} not in allowed list"
            })
        
        # Check for approval if required
        if self.require_approval and self.approval_callback:
            approved = self.approval_callback(action_type, action_data)
            if not approved:
                issues.append({
                    'type': 'approval_denied',
                    'message': f"Action {action_type} was not approved"
                })
        
        return {
            'valid': len(issues) == 0,
            'issues': issues
        }