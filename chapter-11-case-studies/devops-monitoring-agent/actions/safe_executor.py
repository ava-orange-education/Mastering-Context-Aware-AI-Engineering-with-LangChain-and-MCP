"""
Safe Executor

Executes actions with safety checks and validations
"""

from typing import Dict, Any, List, Optional
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class SafeExecutor:
    """
    Safely executes actions with validation and safety checks
    """
    
    def __init__(self):
        # Safety rules
        self.safety_rules = {
            "max_replicas": 100,
            "min_replicas": 1,
            "max_instances": 50,
            "max_concurrent_changes": 3
        }
        
        # Currently executing actions
        self.executing_actions: Dict[str, Dict[str, Any]] = {}
        
        # Dry run mode
        self.dry_run_mode = False
    
    def validate_safety(
        self,
        action: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Validate action safety
        
        Args:
            action: Action to validate
        
        Returns:
            Validation result
        """
        
        validation = {
            "safe": True,
            "warnings": [],
            "blockers": [],
            "risk_level": "low"
        }
        
        action_name = action.get("action")
        parameters = action.get("parameters", {})
        
        # Check concurrent changes
        if len(self.executing_actions) >= self.safety_rules["max_concurrent_changes"]:
            validation["safe"] = False
            validation["blockers"].append(
                f"Too many concurrent changes ({len(self.executing_actions)})"
            )
            validation["risk_level"] = "high"
        
        # Action-specific safety checks
        if action_name == "scale_deployment":
            replicas = parameters.get("replicas", 0)
            
            if replicas > self.safety_rules["max_replicas"]:
                validation["safe"] = False
                validation["blockers"].append(
                    f"Replica count {replicas} exceeds maximum {self.safety_rules['max_replicas']}"
                )
                validation["risk_level"] = "high"
            
            elif replicas < self.safety_rules["min_replicas"]:
                validation["safe"] = False
                validation["blockers"].append(
                    f"Replica count {replicas} below minimum {self.safety_rules['min_replicas']}"
                )
                validation["risk_level"] = "high"
            
            elif replicas == 0:
                validation["warnings"].append("Scaling to 0 will cause downtime")
                validation["risk_level"] = "medium"
        
        elif action_name == "terminate_instance":
            validation["warnings"].append("Instance termination is irreversible")
            validation["risk_level"] = "high"
        
        elif action_name == "modify_security_group":
            validation["warnings"].append("Security group changes affect network access")
            validation["risk_level"] = "high"
        
        elif action_name == "rollback_deployment":
            validation["warnings"].append("Rollback may cause brief service disruption")
            validation["risk_level"] = "medium"
        
        return validation
    
    def check_blast_radius(
        self,
        action: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Estimate blast radius of action
        
        Args:
            action: Action to check
        
        Returns:
            Blast radius assessment
        """
        
        action_name = action.get("action")
        parameters = action.get("parameters", {})
        
        blast_radius = {
            "scope": "single",  # single, service, cluster, region
            "affected_services": [],
            "affected_users": "none",  # none, some, all
            "reversible": True,
            "estimated_impact_time": 0  # seconds
        }
        
        if action_name == "scale_deployment":
            deployment = parameters.get("deployment", "")
            blast_radius["scope"] = "service"
            blast_radius["affected_services"] = [deployment]
            blast_radius["affected_users"] = "some"
            blast_radius["estimated_impact_time"] = 30
        
        elif action_name == "restart_pod":
            pod = parameters.get("pod_name", "")
            blast_radius["scope"] = "single"
            blast_radius["affected_services"] = [pod]
            blast_radius["affected_users"] = "none"
            blast_radius["estimated_impact_time"] = 10
        
        elif action_name == "modify_auto_scaling":
            blast_radius["scope"] = "service"
            blast_radius["affected_users"] = "some"
            blast_radius["estimated_impact_time"] = 60
        
        elif action_name == "rollback_deployment":
            deployment = parameters.get("deployment", "")
            blast_radius["scope"] = "service"
            blast_radius["affected_services"] = [deployment]
            blast_radius["affected_users"] = "all"
            blast_radius["estimated_impact_time"] = 120
            blast_radius["reversible"] = False  # Can't rollback a rollback easily
        
        return blast_radius
    
    async def execute_safely(
        self,
        action: Dict[str, Any],
        executor: Any
    ) -> Dict[str, Any]:
        """
        Execute action with safety checks
        
        Args:
            action: Action to execute
            executor: Action executor
        
        Returns:
            Execution result
        """
        
        action_id = action.get("id", f"action_{datetime.utcnow().timestamp()}")
        
        logger.info(f"Safe execution of: {action.get('action')}")
        
        # Validate safety
        safety_check = self.validate_safety(action)
        
        if not safety_check["safe"]:
            logger.error(f"Safety check failed: {safety_check['blockers']}")
            return {
                "success": False,
                "error": "Safety check failed",
                "blockers": safety_check["blockers"],
                "status": "blocked"
            }
        
        # Check blast radius
        blast_radius = self.check_blast_radius(action)
        
        if blast_radius["scope"] in ["cluster", "region"]:
            logger.warning(f"Large blast radius detected: {blast_radius['scope']}")
        
        # Track execution
        self.executing_actions[action_id] = {
            "action": action,
            "started_at": datetime.utcnow(),
            "status": "running"
        }
        
        try:
            # Execute
            if self.dry_run_mode:
                result = {
                    "success": True,
                    "dry_run": True,
                    "would_execute": action.get("action"),
                    "safety_check": safety_check,
                    "blast_radius": blast_radius
                }
            else:
                # Actual execution through executor
                result = await executor.execute_action(action)
                result["safety_check"] = safety_check
                result["blast_radius"] = blast_radius
            
            # Update status
            self.executing_actions[action_id]["status"] = "completed"
            self.executing_actions[action_id]["completed_at"] = datetime.utcnow()
            
            logger.info(f"Action completed safely: {action_id}")
            
        except Exception as e:
            logger.error(f"Action execution failed: {e}")
            
            self.executing_actions[action_id]["status"] = "failed"
            self.executing_actions[action_id]["error"] = str(e)
            self.executing_actions[action_id]["completed_at"] = datetime.utcnow()
            
            result = {
                "success": False,
                "error": str(e),
                "status": "failed"
            }
        
        finally:
            # Clean up tracking after some time
            # In production, use a background task
            pass
        
        return result
    
    def set_dry_run(self, enabled: bool) -> None:
        """Enable or disable dry-run mode"""
        self.dry_run_mode = enabled
        logger.info(f"Dry-run mode: {'enabled' if enabled else 'disabled'}")
    
    def get_executing_actions(self) -> List[Dict[str, Any]]:
        """Get currently executing actions"""
        return list(self.executing_actions.values())
    
    def update_safety_rule(
        self,
        rule_name: str,
        value: Any
    ) -> None:
        """Update a safety rule"""
        
        if rule_name in self.safety_rules:
            old_value = self.safety_rules[rule_name]
            self.safety_rules[rule_name] = value
            logger.info(f"Updated safety rule {rule_name}: {old_value} -> {value}")
        else:
            logger.warning(f"Unknown safety rule: {rule_name}")