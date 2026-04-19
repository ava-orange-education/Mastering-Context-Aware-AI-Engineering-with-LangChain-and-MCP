"""
Document ACL

Manages Access Control Lists for documents
"""

from typing import List, Dict, Any, Optional, Set
from enum import Enum
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class AccessLevel(Enum):
    """Document access levels"""
    NONE = "none"
    READ = "read"
    WRITE = "write"
    ADMIN = "admin"


class DocumentACL:
    """
    Access Control List for documents
    """
    
    def __init__(self):
        # Document ACLs
        self.acls: Dict[str, Dict[str, Any]] = {}
    
    def set_acl(
        self,
        document_id: str,
        owner: str,
        permissions: Optional[Dict[str, str]] = None
    ) -> None:
        """
        Set ACL for a document
        
        Args:
            document_id: Document identifier
            owner: Document owner
            permissions: User/group permissions
        """
        
        self.acls[document_id] = {
            "owner": owner,
            "created_at": datetime.utcnow().isoformat(),
            "permissions": permissions or {},
            "inherited_from": None
        }
        
        logger.info(f"Set ACL for document {document_id}")
    
    def get_acl(self, document_id: str) -> Optional[Dict[str, Any]]:
        """
        Get ACL for a document
        
        Args:
            document_id: Document identifier
        
        Returns:
            ACL specification or None
        """
        return self.acls.get(document_id)
    
    def check_permission(
        self,
        document_id: str,
        user_id: str,
        required_level: AccessLevel,
        user_groups: Optional[List[str]] = None
    ) -> bool:
        """
        Check if user has required permission level
        
        Args:
            document_id: Document identifier
            user_id: User identifier
            required_level: Required access level
            user_groups: User's group memberships
        
        Returns:
            True if user has permission
        """
        
        acl = self.acls.get(document_id)
        if not acl:
            return False
        
        # Owner has all permissions
        if acl["owner"] == user_id:
            return True
        
        # Check explicit user permissions
        permissions = acl.get("permissions", {})
        user_level_str = permissions.get(f"user:{user_id}")
        
        if user_level_str:
            user_level = AccessLevel(user_level_str)
            if self._has_sufficient_access(user_level, required_level):
                return True
        
        # Check group permissions
        if user_groups:
            for group in user_groups:
                group_level_str = permissions.get(f"group:{group}")
                if group_level_str:
                    group_level = AccessLevel(group_level_str)
                    if self._has_sufficient_access(group_level, required_level):
                        return True
        
        return False
    
    def grant_permission(
        self,
        document_id: str,
        principal: str,
        access_level: AccessLevel,
        principal_type: str = "user"
    ) -> None:
        """
        Grant permission to a user or group
        
        Args:
            document_id: Document identifier
            principal: User or group identifier
            access_level: Access level to grant
            principal_type: "user" or "group"
        """
        
        if document_id not in self.acls:
            raise ValueError(f"ACL not found for document {document_id}")
        
        key = f"{principal_type}:{principal}"
        self.acls[document_id]["permissions"][key] = access_level.value
        
        logger.info(
            f"Granted {access_level.value} to {principal} on {document_id}"
        )
    
    def revoke_permission(
        self,
        document_id: str,
        principal: str,
        principal_type: str = "user"
    ) -> None:
        """
        Revoke permission from a user or group
        
        Args:
            document_id: Document identifier
            principal: User or group identifier
            principal_type: "user" or "group"
        """
        
        if document_id not in self.acls:
            return
        
        key = f"{principal_type}:{principal}"
        permissions = self.acls[document_id].get("permissions", {})
        
        if key in permissions:
            del permissions[key]
            logger.info(f"Revoked access for {principal} on {document_id}")
    
    def get_effective_permissions(
        self,
        document_id: str,
        user_id: str,
        user_groups: Optional[List[str]] = None
    ) -> AccessLevel:
        """
        Get effective permission level for a user
        
        Args:
            document_id: Document identifier
            user_id: User identifier
            user_groups: User's group memberships
        
        Returns:
            Effective access level
        """
        
        acl = self.acls.get(document_id)
        if not acl:
            return AccessLevel.NONE
        
        # Owner has admin access
        if acl["owner"] == user_id:
            return AccessLevel.ADMIN
        
        # Collect all applicable permissions
        levels = []
        permissions = acl.get("permissions", {})
        
        # User permissions
        user_level_str = permissions.get(f"user:{user_id}")
        if user_level_str:
            levels.append(AccessLevel(user_level_str))
        
        # Group permissions
        if user_groups:
            for group in user_groups:
                group_level_str = permissions.get(f"group:{group}")
                if group_level_str:
                    levels.append(AccessLevel(group_level_str))
        
        # Return highest level
        if not levels:
            return AccessLevel.NONE
        
        # Order: ADMIN > WRITE > READ > NONE
        if AccessLevel.ADMIN in levels:
            return AccessLevel.ADMIN
        elif AccessLevel.WRITE in levels:
            return AccessLevel.WRITE
        elif AccessLevel.READ in levels:
            return AccessLevel.READ
        else:
            return AccessLevel.NONE
    
    def list_permissions(self, document_id: str) -> Dict[str, str]:
        """
        List all permissions for a document
        
        Args:
            document_id: Document identifier
        
        Returns:
            Dictionary of principal -> access level
        """
        
        acl = self.acls.get(document_id)
        if not acl:
            return {}
        
        return acl.get("permissions", {})
    
    def _has_sufficient_access(
        self,
        user_level: AccessLevel,
        required_level: AccessLevel
    ) -> bool:
        """Check if user level meets requirement"""
        
        level_hierarchy = {
            AccessLevel.NONE: 0,
            AccessLevel.READ: 1,
            AccessLevel.WRITE: 2,
            AccessLevel.ADMIN: 3
        }
        
        return level_hierarchy[user_level] >= level_hierarchy[required_level]
    
    def inherit_permissions(
        self,
        document_id: str,
        parent_document_id: str
    ) -> None:
        """
        Inherit permissions from parent document
        
        Args:
            document_id: Document identifier
            parent_document_id: Parent document identifier
        """
        
        parent_acl = self.acls.get(parent_document_id)
        if not parent_acl:
            return
        
        # Copy parent permissions
        if document_id not in self.acls:
            self.acls[document_id] = {
                "owner": parent_acl["owner"],
                "created_at": datetime.utcnow().isoformat(),
                "permissions": parent_acl["permissions"].copy(),
                "inherited_from": parent_document_id
            }
        else:
            # Update existing ACL
            self.acls[document_id]["permissions"].update(
                parent_acl["permissions"]
            )
            self.acls[document_id]["inherited_from"] = parent_document_id
        
        logger.info(
            f"Inherited permissions from {parent_document_id} to {document_id}"
        )