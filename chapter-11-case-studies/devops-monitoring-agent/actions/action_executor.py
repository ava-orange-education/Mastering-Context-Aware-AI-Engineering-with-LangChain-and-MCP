"""
Action Executor

Executes remediation actions safely
"""

from typing import Dict, Any, List, Optional
import logging
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class ActionStatus(Enum):
    """Action execution status"""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


class ActionExecutor:
    """
    Executes remediation actions with safety checks
    """
    
    def __init__(self):
        # Action executors by platform
        self.executors = {}
        
        # Execution history
        self.execution_history: List[Dict[str, Any]] = []
        
        # Safety settings
        self.dry_run = False
        self.require_approval = True
        self.max_concurrent_actions = 3
        
        # Circuit breaker
        self.failure_threshold = 5
        self.failure_count = 0
        self.circuit_open = False
    
    def register_executor(
        self,
        platform: str,
        executor: Any
    ) -> None:
        """
        Register an action executor
        
        Args:
            platform: Platform name (kubernetes, aws, etc.)
            executor: Executor instance
        """
        
        self.executors[platform] = executor
        logger.info(f"Registered executor: {platform}")
    
    async def execute_action(
        self,
        action: Dict[str, Any],
        dry_run: Optional[bool] = None
    ) -> Dict[str, Any]:
        """
        Execute a single action
        
        Args:
            action: Action to execute
            dry_run: Override dry_run setting
        
        Returns:
            Execution result
        """
        
        # Check circuit breaker
        if self.circuit_open:
            logger.error("Circuit breaker is open - blocking action execution")
            return {
                "status": ActionStatus.FAILED.value,
                "error": "Circuit breaker is open"
            }
        
        # Determine dry run
        is_dry_run = dry_run if dry_run is not None else self.dry_run
        
        action_name = action.get("action")
        platform = action.get("platform", "kubernetes")
        
        logger.info(f"Executing action: {action_name} (dry_run={is_dry_run})")
        
        # Validate action
        validation = self._validate_action(action)
        if not validation["valid"]:
            logger.error(f"Action validation failed: {validation['errors']}")
            return {
                "status": ActionStatus.FAILED.value,
                "error": f"Validation failed: {validation['errors']}"
            }
        
        # Get executor
        executor = self.executors.get(platform)
        if not executor:
            logger.error(f"No executor for platform: {platform}")
            return {
                "status": ActionStatus.FAILED.value,
                "error": f"No executor for platform: {platform}"
            }
        
        # Create execution record
        execution = {
            "id": f"exec_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
            "action": action_name,
            "platform": platform,
            "dry_run": is_dry_run,
            "started_at": datetime.utcnow(),
            "status": ActionStatus.RUNNING.value,
            "parameters": action.get("parameters", {})
        }
        
        try:
            # Execute action
            if is_dry_run:
                result = await self._dry_run_action(executor, action)
            else:
                result = await self._execute_action_real(executor, action)
            
            # Update execution record
            execution["status"] = ActionStatus.SUCCESS.value
            execution["result"] = result
            execution["completed_at"] = datetime.utcnow()
            
            # Reset failure count on success
            self.failure_count = 0
            
            logger.info(f"Action completed successfully: {action_name}")
            
        except Exception as e:
            logger.error(f"Action execution failed: {e}")
            
            execution["status"] = ActionStatus.FAILED.value
            execution["error"] = str(e)
            execution["completed_at"] = datetime.utcnow()
            
            # Increment failure count
            self.failure_count += 1
            
            # Check circuit breaker threshold
            if self.failure_count >= self.failure_threshold:
                self.circuit_open = True
                logger.error("Circuit breaker opened due to failures")
            
            # Attempt rollback
            if action.get("rollback_plan"):
                await self._rollback_action(execution, action)
        
        # Record execution
        self.execution_history.append(execution)
        
        return execution
    
    async def execute_actions(
        self,
        actions: List[Dict[str, Any]],
        sequential: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Execute multiple actions
        
        Args:
            actions: List of actions to execute
            sequential: Execute sequentially or in parallel
        
        Returns:
            List of execution results
        """
        
        if sequential:
            # Execute one at a time
            results = []
            for action in actions:
                result = await self.execute_action(action)
                results.append(result)
                
                # Stop on failure if required
                if result["status"] == ActionStatus.FAILED.value:
                    if action.get("stop_on_failure", False):
                        logger.warning("Stopping execution due to failure")
                        break
            
            return results
        else:
            # Execute in parallel (with concurrency limit)
            import asyncio
            
            semaphore = asyncio.Semaphore(self.max_concurrent_actions)
            
            async def execute_with_limit(action):
                async with semaphore:
                    return await self.execute_action(action)
            
            tasks = [execute_with_limit(action) for action in actions]
            results = await asyncio.gather(*tasks)
            
            return list(results)
    
    def _validate_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Validate action before execution"""
        
        validation = {
            "valid": True,
            "errors": [],
            "warnings": []
        }
        
        # Check required fields
        if "action" not in action:
            validation["valid"] = False
            validation["errors"].append("Missing 'action' field")
        
        if "platform" not in action:
            validation["warnings"].append("Missing 'platform' field, defaulting to kubernetes")
        
        # Check if action is known
        action_name = action.get("action")
        platform = action.get("platform", "kubernetes")
        
        if platform not in self.executors:
            validation["valid"] = False
            validation["errors"].append(f"Unknown platform: {platform}")
        
        # Platform-specific validation
        if platform in self.executors:
            executor = self.executors[platform]
            if hasattr(executor, 'validate_action'):
                platform_validation = executor.validate_action(action)
                if not platform_validation.get("valid", True):
                    validation["valid"] = False
                    validation["errors"].extend(platform_validation.get("errors", []))
        
        return validation
    
    async def _dry_run_action(
        self,
        executor: Any,
        action: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute action in dry-run mode"""
        
        logger.info("Executing in DRY-RUN mode")
        
        # Simulate execution
        return {
            "dry_run": True,
            "would_execute": action.get("action"),
            "parameters": action.get("parameters", {}),
            "estimated_impact": "Simulated - no actual changes made"
        }
    
    async def _execute_action_real(
        self,
        executor: Any,
        action: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute action for real"""
        
        action_name = action.get("action")
        parameters = action.get("parameters", {})
        
        # Call executor's execute method
        if hasattr(executor, action_name):
            method = getattr(executor, action_name)
            result = await method(**parameters)
            return result
        else:
            raise ValueError(f"Executor does not support action: {action_name}")
    
    async def _rollback_action(
        self,
        execution: Dict[str, Any],
        action: Dict[str, Any]
    ) -> None:
        """Rollback a failed action"""
        
        logger.info(f"Attempting rollback for: {execution['id']}")
        
        rollback_plan = action.get("rollback_plan")
        
        if not rollback_plan:
            logger.warning("No rollback plan available")
            return
        
        try:
            # Execute rollback
            platform = action.get("platform", "kubernetes")
            executor = self.executors.get(platform)
            
            if executor:
                # Rollback is typically the inverse action
                await self._execute_action_real(executor, rollback_plan)
                
                execution["status"] = ActionStatus.ROLLED_BACK.value
                logger.info("Rollback completed successfully")
            
        except Exception as e:
            logger.error(f"Rollback failed: {e}")
            execution["rollback_error"] = str(e)
    
    def get_execution_history(
        self,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get execution history"""
        
        return self.execution_history[-limit:]
    
    def get_execution_stats(self) -> Dict[str, Any]:
        """Get execution statistics"""
        
        total = len(self.execution_history)
        
        if total == 0:
            return {
                "total_executions": 0,
                "success_rate": 0.0,
                "circuit_breaker_open": self.circuit_open
            }
        
        successful = sum(
            1 for e in self.execution_history
            if e["status"] == ActionStatus.SUCCESS.value
        )
        
        failed = sum(
            1 for e in self.execution_history
            if e["status"] == ActionStatus.FAILED.value
        )
        
        return {
            "total_executions": total,
            "successful": successful,
            "failed": failed,
            "rolled_back": total - successful - failed,
            "success_rate": successful / total,
            "failure_rate": failed / total,
            "circuit_breaker_open": self.circuit_open,
            "current_failure_count": self.failure_count
        }
    
    def reset_circuit_breaker(self) -> None:
        """Reset circuit breaker"""
        
        self.circuit_open = False
        self.failure_count = 0
        logger.info("Circuit breaker reset")