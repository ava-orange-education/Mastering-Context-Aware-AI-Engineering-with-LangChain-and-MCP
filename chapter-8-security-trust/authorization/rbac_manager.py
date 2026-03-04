"""
Role-Based Access Control (RBAC) implementation.
"""

from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Permission(Enum):
    """System permissions"""
    # Document permissions
    READ_DOCUMENT = "read_document"
    WRITE_DOCUMENT = "write_document"
    DELETE_DOCUMENT = "delete_document"
    
    # Query permissions
    EXECUTE_QUERY = "execute_query"
    EXPORT_DATA = "export_data"
    
    # Admin permissions
    MANAGE_USERS = "manage_users"
    MANAGE_ROLES = "manage_roles"
    VIEW_AUDIT_LOGS = "view_audit_logs"
    
    # Agent permissions
    USE_AGENT = "use_agent"
    CONFIGURE_AGENT = "configure_agent"
    
    # System permissions
    SYSTEM_ADMIN = "system_admin"


@dataclass
class Role:
    """User role with permissions"""
    role_name: str
    permissions: Set[Permission] = field(default_factory=set)
    description: str = ""
    inherits_from: List[str] = field(default_factory=list)


@dataclass
class Resource:
    """Protected resource"""
    resource_id: str
    resource_type: str
    owner_id: str
    permissions_required: Set[Permission]
    metadata: Dict[str, Any] = field(default_factory=dict)


class RBACManager:
    """Role-Based Access Control Manager"""
    
    def __init__(self):
        """Initialize RBAC manager"""
        self.roles: Dict[str, Role] = {}
        self.user_roles: Dict[str, Set[str]] = {}  # user_id -> role_names
        self.resources: Dict[str, Resource] = {}
        
        # Initialize default roles
        self._initialize_default_roles()
    
    def _initialize_default_roles(self):
        """Create default system roles"""
        # Admin role - full access
        self.create_role(
            role_name="admin",
            permissions=[p for p in Permission],
            description="System administrator with full access"
        )
        
        # User role - basic access
        self.create_role(
            role_name="user",
            permissions=[
                Permission.READ_DOCUMENT,
                Permission.EXECUTE_QUERY,
                Permission.USE_AGENT
            ],
            description="Standard user with read access"
        )
        
        # Analyst role - read and export
        self.create_role(
            role_name="analyst",
            permissions=[
                Permission.READ_DOCUMENT,
                Permission.EXECUTE_QUERY,
                Permission.EXPORT_DATA,
                Permission.USE_AGENT
            ],
            description="Data analyst with read and export capabilities"
        )
        
        # Editor role - read and write
        self.create_role(
            role_name="editor",
            permissions=[
                Permission.READ_DOCUMENT,
                Permission.WRITE_DOCUMENT,
                Permission.EXECUTE_QUERY,
                Permission.USE_AGENT,
                Permission.CONFIGURE_AGENT
            ],
            description="Editor with read/write access"
        )
    
    def create_role(self, role_name: str, permissions: List[Permission],
                   description: str = "", inherits_from: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Create new role
        
        Args:
            role_name: Role name
            permissions: List of permissions
            description: Role description
            inherits_from: Parent roles to inherit from
            
        Returns:
            Creation result
        """
        if role_name in self.roles:
            return {'success': False, 'error': f"Role '{role_name}' already exists"}
        
        # Collect inherited permissions
        all_permissions = set(permissions)
        
        if inherits_from:
            for parent_role_name in inherits_from:
                parent_role = self.roles.get(parent_role_name)
                if parent_role:
                    all_permissions.update(parent_role.permissions)
        
        role = Role(
            role_name=role_name,
            permissions=all_permissions,
            description=description,
            inherits_from=inherits_from or []
        )
        
        self.roles[role_name] = role
        
        logger.info(f"Role created: {role_name} with {len(all_permissions)} permissions")
        
        return {'success': True, 'role_name': role_name}
    
    def assign_role(self, user_id: str, role_name: str) -> Dict[str, Any]:
        """
        Assign role to user
        
        Args:
            user_id: User ID
            role_name: Role name
            
        Returns:
            Assignment result
        """
        if role_name not in self.roles:
            return {'success': False, 'error': f"Role '{role_name}' does not exist"}
        
        if user_id not in self.user_roles:
            self.user_roles[user_id] = set()
        
        self.user_roles[user_id].add(role_name)
        
        logger.info(f"Role '{role_name}' assigned to user {user_id}")
        
        return {'success': True, 'user_id': user_id, 'role': role_name}
    
    def revoke_role(self, user_id: str, role_name: str) -> Dict[str, Any]:
        """
        Revoke role from user
        
        Args:
            user_id: User ID
            role_name: Role name
            
        Returns:
            Revocation result
        """
        if user_id not in self.user_roles:
            return {'success': False, 'error': 'User has no roles'}
        
        if role_name not in self.user_roles[user_id]:
            return {'success': False, 'error': 'User does not have this role'}
        
        self.user_roles[user_id].remove(role_name)
        
        logger.info(f"Role '{role_name}' revoked from user {user_id}")
        
        return {'success': True, 'user_id': user_id, 'role': role_name}
    
    def check_permission(self, user_id: str, permission: Permission) -> bool:
        """
        Check if user has permission
        
        Args:
            user_id: User ID
            permission: Permission to check
            
        Returns:
            True if user has permission
        """
        if user_id not in self.user_roles:
            return False
        
        # Collect all permissions from user's roles
        user_permissions = self._get_user_permissions(user_id)
        
        return permission in user_permissions
    
    def check_resource_access(self, user_id: str, resource_id: str) -> Dict[str, Any]:
        """
        Check if user can access resource
        
        Args:
            user_id: User ID
            resource_id: Resource ID
            
        Returns:
            Access check result
        """
        resource = self.resources.get(resource_id)
        
        if not resource:
            return {'allowed': False, 'reason': 'Resource not found'}
        
        # Owner always has access
        if resource.owner_id == user_id:
            return {'allowed': True, 'reason': 'Owner access'}
        
        # Check required permissions
        user_permissions = self._get_user_permissions(user_id)
        
        missing_permissions = resource.permissions_required - user_permissions
        
        if missing_permissions:
            return {
                'allowed': False,
                'reason': 'Insufficient permissions',
                'missing_permissions': [p.value for p in missing_permissions]
            }
        
        return {'allowed': True, 'reason': 'Permission granted'}
    
    def register_resource(self, resource_id: str, resource_type: str,
                         owner_id: str, permissions_required: List[Permission],
                         metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Register protected resource
        
        Args:
            resource_id: Resource identifier
            resource_type: Type of resource
            owner_id: Resource owner
            permissions_required: Required permissions for access
            metadata: Additional metadata
            
        Returns:
            Registration result
        """
        resource = Resource(
            resource_id=resource_id,
            resource_type=resource_type,
            owner_id=owner_id,
            permissions_required=set(permissions_required),
            metadata=metadata or {}
        )
        
        self.resources[resource_id] = resource
        
        logger.info(f"Resource registered: {resource_id} (type: {resource_type})")
        
        return {'success': True, 'resource_id': resource_id}
    
    def get_user_roles(self, user_id: str) -> List[str]:
        """Get roles assigned to user"""
        return list(self.user_roles.get(user_id, set()))
    
    def get_user_permissions(self, user_id: str) -> List[str]:
        """Get all permissions for user"""
        permissions = self._get_user_permissions(user_id)
        return [p.value for p in permissions]
    
    def _get_user_permissions(self, user_id: str) -> Set[Permission]:
        """Internal: Get user permissions as set"""
        if user_id not in self.user_roles:
            return set()
        
        all_permissions = set()
        
        for role_name in self.user_roles[user_id]:
            role = self.roles.get(role_name)
            if role:
                all_permissions.update(role.permissions)
        
        return all_permissions
    
    def get_role_info(self, role_name: str) -> Optional[Dict[str, Any]]:
        """Get role information"""
        role = self.roles.get(role_name)
        
        if not role:
            return None
        
        return {
            'role_name': role.role_name,
            'description': role.description,
            'permissions': [p.value for p in role.permissions],
            'inherits_from': role.inherits_from,
            'permission_count': len(role.permissions)
        }