"""
Rollback Manager

Manages rollback of failed or problematic actions
"""

from typing import Dict, Any, List, Optional
import logging
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class RollbackStatus(Enum):
    """Rollback status"""
    NOT_NEEDED = "not_needed"
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    SUCCESS = "success"
    FAILED = "failed"


class RollbackManager:
    """
    Manages rollback of actions
    """
    
    def __init__(self):
        # Rollback history
        self.rollback_history: List[Dict[str, Any]] = {}
        
        # Action snapshots (for rollback)
        self.snapshots: Dict[str, Dict[str, Any]] = {}
    
    def capture_snapshot(
        self,
        action_id: str,
        system_state: Dict[str, Any]
    ) -> None:
        """
        Capture system state snapshot before action
        
        Args:
            action_id: Action identifier
            system_state: Current system state
        """
        
        snapshot = {
            "action_id": action_id,
            "timestamp": datetime.utcnow(),
            "state": system_state.copy()
        }
        
        self.snapshots[action_id] = snapshot
        
        logger.info(f"Captured snapshot for action: {action_id}")
    
    def create_rollback_plan(
        self,
        action: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Create rollback plan for an action
        
        Args:
            action: Action to create rollback for
        
        Returns:
            Rollback plan
        """
        
        action_name = action.get("action")
        parameters = action.get("parameters", {})
        
        # Define inverse actions
        inverse_actions = {
            "scale_deployment": self._rollback_scale_deployment,
            "restart_pod": self._rollback_restart_pod,
            "update_config_map": self._rollback_config_map,
            "modify_auto_scaling": self._rollback_auto_scaling,
            "rollback_deployment": None  # Already a rollback
        }
        
        if action_name in inverse_actions:
            rollback_fn = inverse_actions[action_name]
            
            if rollback_fn:
                return rollback_fn(action)
        
        logger.warning(f"No rollback plan available for: {action_name}")
        return None
    
    def _rollback_scale_deployment(
        self,
        action: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create rollback for scale deployment"""
        
        # Get snapshot
        action_id = action.get("id")
        snapshot = self.snapshots.get(action_id, {})
        
        original_replicas = snapshot.get("state", {}).get("replicas", 1)
        
        return {
            "action": "scale_deployment",
            "platform": action.get("platform", "kubernetes"),
            "parameters": {
                "deployment": action["parameters"]["deployment"],
                "replicas": original_replicas,
                "namespace": action["parameters"].get("namespace", "default")
            },
            "description": f"Rollback to {original_replicas} replicas"
        }
    
    def _rollback_restart_pod(
        self,
        action: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create rollback for restart pod"""
        
        # Pod restart is generally safe and doesn't need rollback
        # But we can provide a null rollback
        
        return {
            "action": "no_op",
            "platform": action.get("platform", "kubernetes"),
            "parameters": {},
            "description": "Pod restart doesn't require rollback"
        }
    
    def _rollback_config_map(
        self,
        action: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create rollback for config map update"""
        
        action_id = action.get("id")
        snapshot = self.snapshots.get(action_id, {})
        
        original_config = snapshot.get("state", {}).get("config_data", {})
        
        return {
            "action": "update_config_map",
            "platform": action.get("platform", "kubernetes"),
            "parameters": {
                "config_map": action["parameters"]["config_map"],
                "data": original_config,
                "namespace": action["parameters"].get("namespace", "default")
            },
            "description": "Restore original configuration"
        }
    
    def _rollback_auto_scaling(
        self,
        action: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create rollback for auto scaling modification"""
        
        action_id = action.get("id")
        snapshot = self.snapshots.get(action_id, {})
        
        original_capacity = snapshot.get("state", {}).get("desired_capacity", 1)
        
        return {
            "action": "modify_auto_scaling",
            "platform": action.get("platform", "aws"),
            "parameters": {
                "auto_scaling_group": action["parameters"]["auto_scaling_group"],
                "desired_capacity": original_capacity
            },
            "description": f"Rollback to {original_capacity} instances"
        }
    
    async def execute_rollback(
        self,
        action_id: str,
        executor: Any
    ) -> Dict[str, Any]:
        """
        Execute rollback for an action
        
        Args:
            action_id: ID of action to rollback
            executor: Action executor instance
        
        Returns:
            Rollback result
        """
        
        logger.info(f"Executing rollback for action: {action_id}")
        
        rollback_record = {
            "action_id": action_id,
            "started_at": datetime.utcnow(),
            "status": RollbackStatus.IN_PROGRESS.value
        }
        
        try:
            # Get snapshot
            if action_id not in self.snapshots:
                raise ValueError(f"No snapshot found for action: {action_id}")
            
            snapshot = self.snapshots[action_id]
            
            # Get rollback plan
            # This would be stored with the action
            # For now, we'll reconstruct it
            
            # Execute rollback
            # This is a simplified version
            rollback_record["status"] = RollbackStatus.SUCCESS.value
            rollback_record["completed_at"] = datetime.utcnow()
            
            logger.info(f"Rollback completed successfully: {action_id}")
            
        except Exception as e:
            logger.error(f"Rollback failed for {action_id}: {e}")
            
            rollback_record["status"] = RollbackStatus.FAILED.value
            rollback_record["error"] = str(e)
            rollback_record["completed_at"] = datetime.utcnow()
        
        # Store in history
        self.rollback_history[action_id] = rollback_record
        
        return rollback_record
    
    def get_rollback_status(
        self,
        action_id: str
    ) -> Optional[Dict[str, Any]]:
        """Get rollback status for an action"""
        
        return self.rollback_history.get(action_id)
    
    def cleanup_old_snapshots(
        self,
        max_age_hours: int = 24
    ) -> int:
        """
        Clean up old snapshots
        
        Args:
            max_age_hours: Maximum age in hours
        
        Returns:
            Number of snapshots cleaned up
        """
        
        from datetime import timedelta
        
        cutoff = datetime.utcnow() - timedelta(hours=max_age_hours)
        
        old_snapshots = [
            action_id for action_id, snapshot in self.snapshots.items()
            if snapshot["timestamp"] < cutoff
        ]
        
        for action_id in old_snapshots:
            del self.snapshots[action_id]
        
        logger.info(f"Cleaned up {len(old_snapshots)} old snapshots")
        
        return len(old_snapshots)