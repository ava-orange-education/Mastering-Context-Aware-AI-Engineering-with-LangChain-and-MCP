"""
Organizational Hierarchy

Manages organizational structure and relationships
"""

from typing import Dict, List, Optional, Set
import logging

logger = logging.getLogger(__name__)


class OrganizationalHierarchy:
    """
    Manages organizational hierarchy and relationships
    """
    
    def __init__(self):
        # In-memory organization structure
        # In production, integrate with HR system
        self.departments: Dict[str, Dict[str, Any]] = {}
        self.employees: Dict[str, Dict[str, Any]] = {}
        self.teams: Dict[str, List[str]] = {}
        self.reporting_structure: Dict[str, str] = {}
    
    def add_department(
        self,
        department_id: str,
        name: str,
        parent_department: Optional[str] = None
    ) -> None:
        """
        Add a department to the hierarchy
        
        Args:
            department_id: Department identifier
            name: Department name
            parent_department: Parent department ID
        """
        
        self.departments[department_id] = {
            "name": name,
            "parent": parent_department,
            "subdepartments": []
        }
        
        # Update parent
        if parent_department and parent_department in self.departments:
            self.departments[parent_department]["subdepartments"].append(department_id)
        
        logger.info(f"Added department: {name}")
    
    def add_employee(
        self,
        employee_id: str,
        name: str,
        department_id: str,
        manager_id: Optional[str] = None,
        title: Optional[str] = None
    ) -> None:
        """
        Add an employee to the organization
        
        Args:
            employee_id: Employee identifier
            name: Employee name
            department_id: Department ID
            manager_id: Manager's employee ID
            title: Job title
        """
        
        self.employees[employee_id] = {
            "name": name,
            "department": department_id,
            "manager": manager_id,
            "title": title,
            "direct_reports": []
        }
        
        # Update reporting structure
        if manager_id:
            self.reporting_structure[employee_id] = manager_id
            
            # Update manager's direct reports
            if manager_id in self.employees:
                self.employees[manager_id]["direct_reports"].append(employee_id)
        
        logger.info(f"Added employee: {name} ({department_id})")
    
    def add_team(
        self,
        team_id: str,
        members: List[str]
    ) -> None:
        """
        Add a team
        
        Args:
            team_id: Team identifier
            members: List of employee IDs
        """
        self.teams[team_id] = members
    
    def get_user_department(self, user_id: str) -> Optional[str]:
        """
        Get user's department
        
        Args:
            user_id: User identifier
        
        Returns:
            Department ID or None
        """
        
        employee = self.employees.get(user_id)
        if employee:
            return employee["department"]
        
        return None
    
    def get_department_members(self, department_id: str) -> List[str]:
        """
        Get all members of a department
        
        Args:
            department_id: Department identifier
        
        Returns:
            List of employee IDs
        """
        
        members = []
        
        for employee_id, employee in self.employees.items():
            if employee["department"] == department_id:
                members.append(employee_id)
        
        return members
    
    def get_team_members(self, team_id: str) -> List[str]:
        """
        Get team members
        
        Args:
            team_id: Team identifier
        
        Returns:
            List of employee IDs
        """
        return self.teams.get(team_id, [])
    
    def is_manager_of(
        self,
        manager_id: str,
        employee_id: str,
        include_indirect: bool = False
    ) -> bool:
        """
        Check if one user is manager of another
        
        Args:
            manager_id: Potential manager
            employee_id: Employee to check
            include_indirect: Include indirect reports
        
        Returns:
            True if manager relationship exists
        """
        
        # Check direct report
        employee = self.employees.get(employee_id)
        if employee and employee["manager"] == manager_id:
            return True
        
        # Check indirect reports
        if include_indirect:
            current = employee_id
            visited = set()
            
            while current in self.reporting_structure:
                if current in visited:
                    break  # Circular reference
                
                visited.add(current)
                current = self.reporting_structure[current]
                
                if current == manager_id:
                    return True
        
        return False
    
    def get_reporting_chain(self, employee_id: str) -> List[str]:
        """
        Get full reporting chain for an employee
        
        Args:
            employee_id: Employee identifier
        
        Returns:
            List of manager IDs from direct manager to top
        """
        
        chain = []
        current = employee_id
        visited = set()
        
        while current in self.reporting_structure:
            if current in visited:
                break  # Circular reference
            
            visited.add(current)
            manager = self.reporting_structure[current]
            chain.append(manager)
            current = manager
        
        return chain
    
    def get_direct_reports(self, manager_id: str) -> List[str]:
        """
        Get direct reports for a manager
        
        Args:
            manager_id: Manager identifier
        
        Returns:
            List of direct report employee IDs
        """
        
        manager = self.employees.get(manager_id)
        if manager:
            return manager.get("direct_reports", [])
        
        return []
    
    def get_all_reports(self, manager_id: str) -> List[str]:
        """
        Get all reports (direct and indirect) for a manager
        
        Args:
            manager_id: Manager identifier
        
        Returns:
            List of all report employee IDs
        """
        
        all_reports = []
        to_process = self.get_direct_reports(manager_id)
        visited = set()
        
        while to_process:
            employee_id = to_process.pop(0)
            
            if employee_id in visited:
                continue
            
            visited.add(employee_id)
            all_reports.append(employee_id)
            
            # Add their direct reports
            to_process.extend(self.get_direct_reports(employee_id))
        
        return all_reports
    
    def get_department_hierarchy(self, department_id: str) -> Dict[str, Any]:
        """
        Get department hierarchy
        
        Args:
            department_id: Department identifier
        
        Returns:
            Hierarchical structure
        """
        
        department = self.departments.get(department_id)
        if not department:
            return {}
        
        hierarchy = {
            "id": department_id,
            "name": department["name"],
            "members": self.get_department_members(department_id),
            "subdepartments": []
        }
        
        # Add subdepartments
        for subdept_id in department.get("subdepartments", []):
            subdept_hierarchy = self.get_department_hierarchy(subdept_id)
            if subdept_hierarchy:
                hierarchy["subdepartments"].append(subdept_hierarchy)
        
        return hierarchy