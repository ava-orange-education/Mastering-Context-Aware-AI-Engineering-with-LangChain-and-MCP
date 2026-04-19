"""
Tests for access control
"""

import pytest
import sys
sys.path.append('../..')

from access_control.permission_manager import PermissionManager
from access_control.organizational_hierarchy import OrganizationalHierarchy
from access_control.document_acl import DocumentACL, AccessLevel


class TestPermissionManager:
    """Test Permission Manager"""
    
    @pytest.mark.asyncio
    async def test_set_and_get_permissions(self):
        """Test setting and getting permissions"""
        pm = PermissionManager()
        
        doc_id = "doc_001"
        permissions = {
            "public": False,
            "owner": "user@company.com",
            "groups": ["engineering"],
            "users": ["collaborator@company.com"]
        }
        
        await pm.set_permissions(doc_id, permissions)
        
        retrieved = await pm.get_permissions(doc_id)
        assert retrieved is not None
        assert retrieved["owner"] == "user@company.com"
        assert "engineering" in retrieved["groups"]
    
    @pytest.mark.asyncio
    async def test_check_access_owner(self):
        """Test owner has access"""
        pm = PermissionManager()
        
        doc_id = "doc_002"
        owner = "owner@company.com"
        
        await pm.set_permissions(doc_id, {
            "public": False,
            "owner": owner,
            "groups": [],
            "users": []
        })
        
        has_access = await pm.check_access(owner, doc_id)
        assert has_access is True
    
    @pytest.mark.asyncio
    async def test_check_access_group(self):
        """Test group member has access"""
        pm = PermissionManager()
        
        doc_id = "doc_003"
        user_id = "user@company.com"
        
        await pm.set_permissions(doc_id, {
            "public": False,
            "owner": "owner@company.com",
            "groups": ["engineering"],
            "users": []
        })
        
        pm.set_user_groups(user_id, ["engineering"])
        
        has_access = await pm.check_access(user_id, doc_id, ["engineering"])
        assert has_access is True
    
    @pytest.mark.asyncio
    async def test_check_access_denied(self):
        """Test access denied for unauthorized user"""
        pm = PermissionManager()
        
        doc_id = "doc_004"
        user_id = "user@company.com"
        
        await pm.set_permissions(doc_id, {
            "public": False,
            "owner": "owner@company.com",
            "groups": ["sales"],
            "users": []
        })
        
        pm.set_user_groups(user_id, ["engineering"])
        
        has_access = await pm.check_access(user_id, doc_id, ["engineering"])
        assert has_access is False
    
    @pytest.mark.asyncio
    async def test_grant_access(self):
        """Test granting access to user"""
        pm = PermissionManager()
        
        doc_id = "doc_005"
        user_id = "user@company.com"
        
        await pm.set_permissions(doc_id, {
            "public": False,
            "owner": "owner@company.com",
            "groups": [],
            "users": []
        })
        
        await pm.grant_access(doc_id, user_id)
        
        has_access = await pm.check_access(user_id, doc_id)
        assert has_access is True


class TestOrganizationalHierarchy:
    """Test Organizational Hierarchy"""
    
    def test_add_department(self):
        """Test adding departments"""
        org = OrganizationalHierarchy()
        
        org.add_department("dept_eng", "Engineering")
        org.add_department("dept_prod", "Product", parent_department="dept_eng")
        
        assert "dept_eng" in org.departments
        assert "dept_prod" in org.departments
        assert org.departments["dept_prod"]["parent"] == "dept_eng"
    
    def test_add_employee(self):
        """Test adding employees"""
        org = OrganizationalHierarchy()
        
        org.add_department("dept_eng", "Engineering")
        org.add_employee("emp_001", "John Doe", "dept_eng", title="Engineer")
        org.add_employee("emp_002", "Jane Smith", "dept_eng", manager_id="emp_001", title="Senior Engineer")
        
        assert "emp_001" in org.employees
        assert org.employees["emp_002"]["manager"] == "emp_001"
    
    def test_get_department_members(self):
        """Test getting department members"""
        org = OrganizationalHierarchy()
        
        org.add_department("dept_eng", "Engineering")
        org.add_employee("emp_001", "John", "dept_eng")
        org.add_employee("emp_002", "Jane", "dept_eng")
        
        members = org.get_department_members("dept_eng")
        assert len(members) == 2
        assert "emp_001" in members
        assert "emp_002" in members
    
    def test_is_manager_of(self):
        """Test manager relationship"""
        org = OrganizationalHierarchy()
        
        org.add_department("dept_eng", "Engineering")
        org.add_employee("emp_mgr", "Manager", "dept_eng")
        org.add_employee("emp_001", "Employee", "dept_eng", manager_id="emp_mgr")
        
        assert org.is_manager_of("emp_mgr", "emp_001") is True
        assert org.is_manager_of("emp_001", "emp_mgr") is False
    
    def test_get_reporting_chain(self):
        """Test getting reporting chain"""
        org = OrganizationalHierarchy()
        
        org.add_department("dept_eng", "Engineering")
        org.add_employee("emp_ceo", "CEO", "dept_eng")
        org.add_employee("emp_vp", "VP", "dept_eng", manager_id="emp_ceo")
        org.add_employee("emp_mgr", "Manager", "dept_eng", manager_id="emp_vp")
        org.add_employee("emp_001", "Employee", "dept_eng", manager_id="emp_mgr")
        
        chain = org.get_reporting_chain("emp_001")
        assert len(chain) == 3
        assert chain[0] == "emp_mgr"
        assert chain[1] == "emp_vp"
        assert chain[2] == "emp_ceo"


class TestDocumentACL:
    """Test Document ACL"""
    
    def test_set_and_get_acl(self):
        """Test setting and getting ACL"""
        acl = DocumentACL()
        
        doc_id = "doc_001"
        owner = "user@company.com"
        
        acl.set_acl(doc_id, owner, {
            "user:collaborator@company.com": "write",
            "group:engineering": "read"
        })
        
        retrieved = acl.get_acl(doc_id)
        assert retrieved is not None
        assert retrieved["owner"] == owner
    
    def test_check_permission_owner(self):
        """Test owner has all permissions"""
        acl = DocumentACL()
        
        doc_id = "doc_002"
        owner = "owner@company.com"
        
        acl.set_acl(doc_id, owner)
        
        assert acl.check_permission(doc_id, owner, AccessLevel.READ) is True
        assert acl.check_permission(doc_id, owner, AccessLevel.WRITE) is True
        assert acl.check_permission(doc_id, owner, AccessLevel.ADMIN) is True
    
    def test_check_permission_user(self):
        """Test user permissions"""
        acl = DocumentACL()
        
        doc_id = "doc_003"
        owner = "owner@company.com"
        user = "user@company.com"
        
        acl.set_acl(doc_id, owner, {
            "user:user@company.com": "read"
        })
        
        assert acl.check_permission(doc_i