"""
Access Control

Role-based access control (RBAC) for patient data
"""

from typing import Dict, List, Optional, Set
from enum import Enum
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class Role(Enum):
    """User roles"""
    PATIENT = "patient"
    PHYSICIAN = "physician"
    NURSE = "nurse"
    GENETIC_COUNSELOR = "genetic_counselor"
    RESEARCHER = "researcher"
    ADMIN = "admin"


class Permission(Enum):
    """Permissions"""
    VIEW_GENETIC_DATA = "view_genetic_data"
    VIEW_CLINICAL_DATA = "view_clinical_data"
    VIEW_DEMOGRAPHICS = "view_demographics"
    EDIT_GENETIC_DATA = "edit_genetic_data"
    EDIT_CLINICAL_DATA = "edit_clinical_data"
    GENERATE_RECOMMENDATIONS = "generate_recommendations"
    VIEW_AUDIT_LOGS = "view_audit_logs"
    MANAGE_USERS = "manage_users"
    DELETE_DATA = "delete_data"


class AccessControl:
    """
    Role-based access control system
    """
    
    def __init__(self):
        # Define role permissions
        self.role_permissions: Dict[Role, Set[Permission]] = {
            Role.PATIENT: {
                Permission.VIEW_GENETIC_DATA,
                Permission.VIEW_CLINICAL_DATA,
                Permission.VIEW_DEMOGRAPHICS,
            },
            Role.PHYSICIAN: {
                Permission.VIEW_GENETIC_DATA,
                Permission.VIEW_CLINICAL_DATA,
                Permission.VIEW_DEMOGRAPHICS,
                Permission.EDIT_CLINICAL_DATA,
                Permission.GENERATE_RECOMMENDATIONS,
            },
            Role.NURSE: {
                Permission.VIEW_DEMOGRAPHICS,
                Permission.VIEW_CLINICAL_DATA,
            },
            Role.GENETIC_COUNSELOR: {
                Permission.VIEW_GENETIC_DATA,
                Permission.VIEW_CLINICAL_DATA,
                Permission.VIEW_DEMOGRAPHICS,
                Permission.EDIT_GENETIC_DATA,
                Permission.GENERATE_RECOMMENDATIONS,
            },
            Role.RESEARCHER: {
                Permission.VIEW_GENETIC_DATA,  # De-identified only
                Permission.VIEW_CLINICAL_DATA,  # De-identified only
            },
            Role.ADMIN: set(Permission),  # All permissions
        }
        
        # Track access grants
        self.access_grants: Dict[str, Dict[str, datetime]] = {}
    
    def check_permission(
        self,
        user_id: str,
        user_role: Role,
        permission: Permission,
        patient_id: str
    ) -> bool:
        """
        Check if user has permission
        
        Args:
            user_id: User ID
            user_role: User's role
            permission: Permission to check
            patient_id: Patient ID being accessed
        
        Returns:
            True if permission granted
        """
        
        # Check if role has permission
        if permission not in self.role_permissions.get(user_role, set()):
            logger.warning(
                f"Permission denied: {user_id} ({user_role.value}) "
                f"lacks {permission.value}"
            )
            return False
        
        # Patients can only access their own data
        if user_role == Role.PATIENT and user_id != patient_id:
            logger.warning(
                f"Permission denied: Patient {user_id} "
                f"attempted to access {patient_id}'s data"
            )
            return False
        
        # Check emergency access grants
        if self._has_emergency_access(user_id, patient_id):
            logger.info(f"Emergency access granted: {user_id} -> {patient_id}")
            return True
        
        return True
    
    def grant_emergency_access(
        self,
        user_id: str,
        patient_id: str,
        duration_minutes: int = 60
    ) -> None:
        """
        Grant emergency access (break-glass)
        
        Args:
            user_id: User to grant access
            patient_id: Patient data to access
            duration_minutes: Duration of access grant
        """
        
        if user_id not in self.access_grants:
            self.access_grants[user_id] = {}
        
        expires_at = datetime.utcnow()
        # Add duration (simplified - use timedelta in production)
        
        self.access_grants[user_id][patient_id] = expires_at
        
        logger.warning(
            f"EMERGENCY ACCESS GRANTED: {user_id} -> {patient_id} "
            f"(expires in {duration_minutes} min)"
        )
    
    def _has_emergency_access(
        self,
        user_id: str,
        patient_id: str
    ) -> bool:
        """Check if user has active emergency access"""
        
        if user_id not in self.access_grants:
            return False
        
        if patient_id not in self.access_grants[user_id]:
            return False
        
        # Check expiration
        expires_at = self.access_grants[user_id][patient_id]
        
        if datetime.utcnow() > expires_at:
            # Access expired
            del self.access_grants[user_id][patient_id]
            return False
        
        return True
    
    def get_user_permissions(self, user_role: Role) -> List[Permission]:
        """
        Get all permissions for a role
        
        Args:
            user_role: User role
        
        Returns:
            List of permissions
        """
        return list(self.role_permissions.get(user_role, set()))
    
    def validate_data_access(
        self,
        user_id: str,
        user_role: Role,
        patient_id: str,
        data_types: List[str]
    ) -> Dict[str, bool]:
        """
        Validate access to specific data types
        
        Args:
            user_id: User ID
            user_role: User role
            patient_id: Patient ID
            data_types: Types of data being accessed
        
        Returns:
            Dict mapping data type to access allowed
        """
        
        # Map data types to permissions
        data_type_permissions = {
            "genetic_data": Permission.VIEW_GENETIC_DATA,
            "clinical_data": Permission.VIEW_CLINICAL_DATA,
            "demographics": Permission.VIEW_DEMOGRAPHICS,
        }
        
        access_results = {}
        
        for data_type in data_types:
            permission = data_type_permissions.get(data_type)
            
            if permission:
                access_results[data_type] = self.check_permission(
                    user_id, user_role, permission, patient_id
                )
            else:
                access_results[data_type] = False
        
        return access_results