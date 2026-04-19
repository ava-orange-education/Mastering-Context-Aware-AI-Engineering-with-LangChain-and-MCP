"""
Permission Manager

Manages document access permissions and authorization
"""

from typing import List, Dict, Any, Optional
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class PermissionManager:
    """
    Manages document permissions and access control
    """
    
    def __init__(self):
        # In-memory permission storage
        # In production, use database
        self.document_permissions: Dict[str, Dict[str, Any]] = {}
        self.user_groups: Dict[str, List[str]] = {}
    
    async def check_access(
        self,
        user_id: str,
        document_id: str,
        user_groups: Optional[List[str]] = None
    ) -> bool:
        """
        Check if user has access to document
        
        Args:
            user_id: User identifier
            document_id: Document identifier
            user_groups: User's group memberships
        
        Returns:
            True if user has access
        """
        
        # Get document permissions
        permissions = self.document_permissions.get(document_id)
        
        if not permissions:
            # No permissions set - default to deny
            logger.debug(f"No permissions found for document {document_id}")
            return False
        
        # Check if document is public
        if permissions.get("public", False):
            return True
        
        # Check if user is owner
        if permissions.get("owner") == user_id:
            return True
        
        # Check if user is in allowed users list
        allowed_users = permissions.get("users", [])
        if user_id in allowed_users:
            return True
        
        # Check if user is in allowed groups
        allowed_groups = permissions.get("groups", [])
        user_groups = user_groups or self.user_groups.get(user_id, [])
        
        if any(group in allowed_groups for group in user_groups):
            return True
        
        # Check department access
        user_department = self._get_user_department(user_id)
        allowed_departments = permissions.get("departments", [])
        
        if user_department and user_department in allowed_departments:
            return True
        
        # No access granted
        logger.debug(f"Access denied: {user_id} to {document_id}")
        return False
    
    async def set_permissions(
        self,
        document_id: str,
        permissions: Dict[str, Any]
    ) -> None:
        """
        Set permissions for a document
        
        Args:
            document_id: Document identifier
            permissions: Permission specification
        """
        
        self.document_permissions[document_id] = {
            "public": permissions.get("public", False),
            "owner": permissions.get("owner"),
            "users": permissions.get("users", []),
            "groups": permissions.get("groups", []),
            "departments": permissions.get("departments", []),
            "updated_at": datetime.utcnow().isoformat()
        }
        
        logger.info(f"Set permissions for document {document_id}")
    
    async def get_permissions(self, document_id: str) -> Optional[Dict[str, Any]]:
        """
        Get permissions for a document
        
        Args:
            document_id: Document identifier
        
        Returns:
            Permission specification or None
        """
        return self.document_permissions.get(document_id)
    
    async def grant_access(
        self,
        document_id: str,
        user_id: str
    ) -> None:
        """
        Grant access to a specific user
        
        Args:
            document_id: Document identifier
            user_id: User to grant access
        """
        
        if document_id not in self.document_permissions:
            self.document_permissions[document_id] = {
                "public": False,
                "users": [],
                "groups": [],
                "departments": []
            }
        
        users = self.document_permissions[document_id].get("users", [])
        if user_id not in users:
            users.append(user_id)
            self.document_permissions[document_id]["users"] = users
        
        logger.info(f"Granted access: {user_id} to {document_id}")
    
    async def revoke_access(
        self,
        document_id: str,
        user_id: str
    ) -> None:
        """
        Revoke access from a specific user
        
        Args:
            document_id: Document identifier
            user_id: User to revoke access
        """
        
        if document_id in self.document_permissions:
            users = self.document_permissions[document_id].get("users", [])
            if user_id in users:
                users.remove(user_id)
                self.document_permissions[document_id]["users"] = users
        
        logger.info(f"Revoked access: {user_id} from {document_id}")
    
    async def grant_group_access(
        self,
        document_id: str,
        group_name: str
    ) -> None:
        """
        Grant access to a group
        
        Args:
            document_id: Document identifier
            group_name: Group to grant access
        """
        
        if document_id not in self.document_permissions:
            self.document_permissions[document_id] = {
                "public": False,
                "users": [],
                "groups": [],
                "departments": []
            }
        
        groups = self.document_permissions[document_id].get("groups", [])
        if group_name not in groups:
            groups.append(group_name)
            self.document_permissions[document_id]["groups"] = groups
        
        logger.info(f"Granted group access: {group_name} to {document_id}")
    
    async def get_accessible_documents(
        self,
        user_id: str,
        document_ids: List[str],
        user_groups: Optional[List[str]] = None
    ) -> List[str]:
        """
        Filter document IDs to only accessible ones
        
        Args:
            user_id: User identifier
            document_ids: List of document IDs to check
            user_groups: User's group memberships
        
        Returns:
            List of accessible document IDs
        """
        
        accessible = []
        
        for doc_id in document_ids:
            has_access = await self.check_access(
                user_id=user_id,
                document_id=doc_id,
                user_groups=user_groups
            )
            
            if has_access:
                accessible.append(doc_id)
        
        return accessible
    
    def set_user_groups(
        self,
        user_id: str,
        groups: List[str]
    ) -> None:
        """
        Set user's group memberships
        
        Args:
            user_id: User identifier
            groups: List of group names
        """
        self.user_groups[user_id] = groups
    
    def get_user_groups(self, user_id: str) -> List[str]:
        """
        Get user's group memberships
        
        Args:
            user_id: User identifier
        
        Returns:
            List of group names
        """
        return self.user_groups.get(user_id, [])
    
    def _get_user_department(self, user_id: str) -> Optional[str]:
        """
        Get user's department
        
        In production, integrate with HR system or Active Directory
        """
        
        # Simplified - extract from email domain or lookup
        if "@" in user_id:
            # Try to infer from email
            if "engineering@" in user_id:
                return "engineering"
            elif "sales@" in user_id:
                return "sales"
            elif "marketing@" in user_id:
                return "marketing"
        
        return None