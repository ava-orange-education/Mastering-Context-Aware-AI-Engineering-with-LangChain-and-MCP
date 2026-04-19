"""
Approval Workflow

Manages approval process for high-risk actions
"""

from typing import Dict, Any, List, Optional
import logging
from datetime import datetime, timedelta
from enum import Enum

logger = logging.getLogger(__name__)


class ApprovalStatus(Enum):
    """Approval status"""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"


class ApprovalWorkflow:
    """
    Manages approval workflow for high-risk actions
    """
    
    def __init__(self):
        # Pending approvals
        self.pending_approvals: Dict[str, Dict[str, Any]] = {}
        
        # Approval history
        self.approval_history: List[Dict[str, Any]] = []
        
        # Approval settings
        self.approval_timeout = 3600  # 1 hour
        self.require_multiple_approvers = False
        self.min_approvers = 2
        
        # Approvers by role
        self.approvers: Dict[str, List[str]] = {
            "admin": [],
            "operator": [],
            "manager": []
        }
    
    def register_approver(
        self,
        user_id: str,
        role: str = "operator"
    ) -> None:
        """
        Register an approver
        
        Args:
            user_id: User identifier
            role: User role (admin, operator, manager)
        """
        
        if role not in self.approvers:
            self.approvers[role] = []
        
        if user_id not in self.approvers[role]:
            self.approvers[role].append(user_id)
            logger.info(f"Registered approver: {user_id} (role: {role})")
    
    def request_approval(
        self,
        action: Dict[str, Any],
        requester: str,
        reason: str
    ) -> str:
        """
        Request approval for an action
        
        Args:
            action: Action requiring approval
            requester: User requesting the action
            reason: Reason for the action
        
        Returns:
            Approval request ID
        """
        
        request_id = f"approval_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
        
        approval_request = {
            "id": request_id,
            "action": action,
            "requester": requester,
            "reason": reason,
            "status": ApprovalStatus.PENDING.value,
            "created_at": datetime.utcnow(),
            "expires_at": datetime.utcnow() + timedelta(seconds=self.approval_timeout),
            "approvals": [],
            "rejections": []
        }
        
        self.pending_approvals[request_id] = approval_request
        
        logger.info(f"Approval requested: {request_id} by {requester}")
        
        # Notify approvers
        self._notify_approvers(approval_request)
        
        return request_id
    
    def approve(
        self,
        request_id: str,
        approver: str,
        comment: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Approve an action
        
        Args:
            request_id: Approval request ID
            approver: User approving
            comment: Optional approval comment
        
        Returns:
            Approval result
        """
        
        if request_id not in self.pending_approvals:
            return {
                "success": False,
                "error": "Approval request not found"
            }
        
        request = self.pending_approvals[request_id]
        
        # Check if expired
        if datetime.utcnow() > request["expires_at"]:
            request["status"] = ApprovalStatus.EXPIRED.value
            return {
                "success": False,
                "error": "Approval request expired"
            }
        
        # Check if approver is authorized
        if not self._is_authorized_approver(approver):
            return {
                "success": False,
                "error": "User not authorized to approve"
            }
        
        # Check if already approved by this user
        if approver in [a["approver"] for a in request["approvals"]]:
            return {
                "success": False,
                "error": "Already approved by this user"
            }
        
        # Add approval
        request["approvals"].append({
            "approver": approver,
            "timestamp": datetime.utcnow(),
            "comment": comment
        })
        
        logger.info(f"Approval granted by {approver} for {request_id}")
        
        # Check if enough approvals
        if self._has_sufficient_approvals(request):
            request["status"] = ApprovalStatus.APPROVED.value
            
            # Move to history
            self.approval_history.append(request.copy())
            del self.pending_approvals[request_id]
            
            logger.info(f"Action approved: {request_id}")
            
            return {
                "success": True,
                "status": "approved",
                "message": "Action approved and ready for execution"
            }
        else:
            needed = self.min_approvers - len(request["approvals"])
            return {
                "success": True,
                "status": "pending",
                "message": f"Approval recorded. {needed} more approval(s) needed."
            }
    
    def reject(
        self,
        request_id: str,
        rejector: str,
        reason: str
    ) -> Dict[str, Any]:
        """
        Reject an action
        
        Args:
            request_id: Approval request ID
            rejector: User rejecting
            reason: Rejection reason
        
        Returns:
            Rejection result
        """
        
        if request_id not in self.pending_approvals:
            return {
                "success": False,
                "error": "Approval request not found"
            }
        
        request = self.pending_approvals[request_id]
        
        # Add rejection
        request["rejections"].append({
            "rejector": rejector,
            "timestamp": datetime.utcnow(),
            "reason": reason
        })
        
        request["status"] = ApprovalStatus.REJECTED.value
        
        logger.info(f"Action rejected by {rejector}: {request_id}")
        
        # Move to history
        self.approval_history.append(request.copy())
        del self.pending_approvals[request_id]
        
        return {
            "success": True,
            "status": "rejected",
            "message": f"Action rejected: {reason}"
        }
    
    def get_approval_status(
        self,
        request_id: str
    ) -> Optional[Dict[str, Any]]:
        """Get status of approval request"""
        
        # Check pending
        if request_id in self.pending_approvals:
            request = self.pending_approvals[request_id]
            
            # Check if expired
            if datetime.utcnow() > request["expires_at"]:
                request["status"] = ApprovalStatus.EXPIRED.value
                self.approval_history.append(request.copy())
                del self.pending_approvals[request_id]
            
            return request
        
        # Check history
        for request in self.approval_history:
            if request["id"] == request_id:
                return request
        
        return None
    
    def _is_authorized_approver(self, user_id: str) -> bool:
        """Check if user is authorized to approve"""
        
        for role, users in self.approvers.items():
            if user_id in users:
                return True
        
        return False
    
    def _has_sufficient_approvals(self, request: Dict[str, Any]) -> bool:
        """Check if request has sufficient approvals"""
        
        if self.require_multiple_approvers:
            return len(request["approvals"]) >= self.min_approvers
        else:
            return len(request["approvals"]) >= 1
    
    def _notify_approvers(self, request: Dict[str, Any]) -> None:
        """Notify approvers of pending request"""
        
        # Get all approvers
        all_approvers = []
        for users in self.approvers.values():
            all_approvers.extend(users)
        
        logger.info(f"Notifying {len(all_approvers)} approvers for {request['id']}")
        
        # In production, send actual notifications (email, Slack, etc.)
    
    def cleanup_expired(self) -> int:
        """Clean up expired approval requests"""
        
        expired_count = 0
        expired_ids = []
        
        for request_id, request in self.pending_approvals.items():
            if datetime.utcnow() > request["expires_at"]:
                request["status"] = ApprovalStatus.EXPIRED.value
                self.approval_history.append(request.copy())
                expired_ids.append(request_id)
                expired_count += 1
        
        # Remove expired
        for request_id in expired_ids:
            del self.pending_approvals[request_id]
        
        if expired_count > 0:
            logger.info(f"Cleaned up {expired_count} expired approval requests")
        
        return expired_count
    
    def get_pending_approvals(
        self,
        user_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get pending approval requests"""
        
        # Clean up expired first
        self.cleanup_expired()
        
        pending = list(self.pending_approvals.values())
        
        # Filter by user if specified
        if user_id:
            # Return approvals that this user can act on
            if self._is_authorized_approver(user_id):
                # Filter out ones already approved by this user
                pending = [
                    r for r in pending
                    if user_id not in [a["approver"] for a in r["approvals"]]
                ]
        
        return pending
    
    def get_approval_stats(self) -> Dict[str, Any]:
        """Get approval statistics"""
        
        total_requests = len(self.approval_history) + len(self.pending_approvals)
        
        approved = sum(
            1 for r in self.approval_history
            if r["status"] == ApprovalStatus.APPROVED.value
        )
        
        rejected = sum(
            1 for r in self.approval_history
            if r["status"] == ApprovalStatus.REJECTED.value
        )
        
        expired = sum(
            1 for r in self.approval_history
            if r["status"] == ApprovalStatus.EXPIRED.value
        )
        
        return {
            "total_requests": total_requests,
            "pending": len(self.pending_approvals),
            "approved": approved,
            "rejected": rejected,
            "expired": expired,
            "approval_rate": approved / total_requests if total_requests > 0 else 0
        }