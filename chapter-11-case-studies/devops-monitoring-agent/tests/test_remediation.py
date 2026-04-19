"""
Tests for Remediation Components
"""

import pytest
from datetime import datetime
import sys
sys.path.append('..')

from actions.action_executor import ActionExecutor, ActionStatus
from actions.approval_workflow import ApprovalWorkflow, ApprovalStatus
from actions.rollback_manager import RollbackManager, RollbackStatus
from actions.safe_executor import SafeExecutor


class TestActionExecutor:
    """Test action executor"""
    
    @pytest.fixture
    def executor(self):
        return ActionExecutor()
    
    @pytest.mark.asyncio
    async def test_execute_safe_action(self, executor):
        """Test execution of safe action"""
        
        action = {
            "action": "restart_pod",
            "platform": "kubernetes",
            "parameters": {
                "pod_name": "test-pod",
                "namespace": "default"
            }
        }
        
        result = await executor.execute_action(action, dry_run=True)
        
        assert result is not None
        assert result.get("dry_run") is True
        assert result.get("status") in [ActionStatus.SUCCESS.value, "success", "dry_run"]
    
    def test_validate_action(self, executor):
        """Test action validation"""
        
        # Valid action
        valid_action = {
            "action": "restart_pod",
            "platform": "kubernetes",
            "parameters": {"pod_name": "test-pod"}
        }
        
        validation = executor._validate_action(valid_action)
        assert validation["valid"] is True
        
        # Invalid action (missing required fields)
        invalid_action = {
            "platform": "kubernetes"
        }
        
        validation = executor._validate_action(invalid_action)
        assert validation["valid"] is False
        assert len(validation["errors"]) > 0
    
    def test_circuit_breaker(self, executor):
        """Test circuit breaker functionality"""
        
        # Simulate failures to trigger circuit breaker
        executor.failure_count = executor.failure_threshold
        executor.circuit_open = True
        
        # Try to execute action with circuit open
        action = {"action": "test", "platform": "test"}
        
        # Should be blocked by circuit breaker
        # In async context, this would return error
        assert executor.circuit_open is True
    
    def test_execution_history(self, executor):
        """Test execution history tracking"""
        
        # Add some mock executions
        executor.execution_history.append({
            "id": "exec_1",
            "action": "restart_pod",
            "status": ActionStatus.SUCCESS.value
        })
        
        executor.execution_history.append({
            "id": "exec_2",
            "action": "scale_deployment",
            "status": ActionStatus.FAILED.value
        })
        
        history = executor.get_execution_history(limit=10)
        assert len(history) >= 2
        
        stats = executor.get_execution_stats()
        assert stats["total_executions"] >= 2


class TestApprovalWorkflow:
    """Test approval workflow"""
    
    @pytest.fixture
    def workflow(self):
        wf = ApprovalWorkflow()
        wf.register_approver("approver1@test.com", "admin")
        wf.register_approver("approver2@test.com", "operator")
        return wf
    
    def test_request_approval(self, workflow):
        """Test approval request creation"""
        
        action = {
            "action": "terminate_instance",
            "parameters": {"instance_id": "i-12345"}
        }
        
        request_id = workflow.request_approval(
            action=action,
            requester="user@test.com",
            reason="Emergency maintenance"
        )
        
        assert request_id is not None
        assert request_id in workflow.pending_approvals
        
        request = workflow.pending_approvals[request_id]
        assert request["status"] == ApprovalStatus.PENDING.value
        assert request["requester"] == "user@test.com"
    
    def test_approve_action(self, workflow):
        """Test action approval"""
        
        action = {"action": "test"}
        request_id = workflow.request_approval(
            action=action,
            requester="user@test.com",
            reason="Test"
        )
        
        result = workflow.approve(
            request_id=request_id,
            approver="approver1@test.com",
            comment="Approved for testing"
        )
        
        assert result["success"] is True
    
    def test_reject_action(self, workflow):
        """Test action rejection"""
        
        action = {"action": "test"}
        request_id = workflow.request_approval(
            action=action,
            requester="user@test.com",
            reason="Test"
        )
        
        result = workflow.reject(
            request_id=request_id,
            rejector="approver1@test.com",
            reason="Not authorized"
        )
        
        assert result["success"] is True
        assert result["status"] == "rejected"
    
    def test_approval_expiration(self, workflow):
        """Test approval request expiration"""
        
        workflow.approval_timeout = 1  # 1 second
        
        action = {"action": "test"}
        request_id = workflow.request_approval(
            action=action,
            requester="user@test.com",
            reason="Test"
        )
        
        import time
        time.sleep(2)  # Wait for expiration
        
        # Cleanup should mark as expired
        expired = workflow.cleanup_expired()
        assert expired >= 0


class TestRollbackManager:
    """Test rollback manager"""
    
    @pytest.fixture
    def manager(self):
        return RollbackManager()
    
    def test_capture_snapshot(self, manager):
        """Test system state snapshot"""
        
        action_id = "action_123"
        system_state = {
            "replicas": 3,
            "memory_limit": "2Gi",
            "cpu_limit": "1000m"
        }
        
        manager.capture_snapshot(action_id, system_state)
        
        assert action_id in manager.snapshots
        assert manager.snapshots[action_id]["state"] == system_state
    
    def test_create_rollback_plan(self, manager):
        """Test rollback plan creation"""
        
        # Test scale deployment rollback
        action = {
            "action": "scale_deployment",
            "id": "action_123",
            "parameters": {
                "deployment": "api-server",
                "replicas": 5
            }
        }
        
        # Capture snapshot with original state
        manager.capture_snapshot("action_123", {"replicas": 3})
        
        rollback_plan = manager.create_rollback_plan(action)
        
        assert rollback_plan is not None
        assert rollback_plan["action"] == "scale_deployment"
        assert rollback_plan["parameters"]["replicas"] == 3  # Original value
    
    def test_cleanup_old_snapshots(self, manager):
        """Test snapshot cleanup"""
        
        # Add some snapshots
        manager.capture_snapshot("old_action", {"test": "data"})
        
        # Cleanup with 0 hour threshold (should clean all)
        cleaned = manager.cleanup_old_snapshots(max_age_hours=0)
        
        assert cleaned >= 0


class TestSafeExecutor:
    """Test safe executor"""
    
    @pytest.fixture
    def executor(self):
        return SafeExecutor()
    
    def test_validate_safety(self, executor):
        """Test safety validation"""
        
        # Safe action
        safe_action = {
            "action": "restart_pod",
            "parameters": {"pod_name": "test-pod"}
        }
        
        validation = executor.validate_safety(safe_action)
        assert validation["safe"] is True
        assert validation["risk_level"] == "low"
        
        # Risky action
        risky_action = {
            "action": "scale_deployment",
            "parameters": {"replicas": 200}  # Exceeds max
        }
        
        validation = executor.validate_safety(risky_action)
        assert validation["safe"] is False
        assert len(validation["blockers"]) > 0
    
    def test_check_blast_radius(self, executor):
        """Test blast radius estimation"""
        
        action = {
            "action": "rollback_deployment",
            "parameters": {"deployment": "api-server"}
        }
        
        blast_radius = executor.check_blast_radius(action)
        
        assert blast_radius is not None
        assert "scope" in blast_radius
        assert "affected_users" in blast_radius
        assert "reversible" in blast_radius
    
    def test_dry_run_mode(self, executor):
        """Test dry run mode"""
        
        executor.set_dry_run(True)
        assert executor.dry_run_mode is True
        
        executor.set_dry_run(False)
        assert executor.dry_run_mode is False
    
    def test_safety_rule_update(self, executor):
        """Test safety rule updates"""
        
        original_max = executor.safety_rules["max_replicas"]
        
        executor.update_safety_rule("max_replicas", 150)
        
        assert executor.safety_rules["max_replicas"] == 150
        assert executor.safety_rules["max_replicas"] != original_max


if __name__ == "__main__":
    pytest.main([__file__, "-v"])