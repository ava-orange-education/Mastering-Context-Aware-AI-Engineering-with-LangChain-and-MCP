"""
Example 2: RBAC Implementation
Demonstrates role-based access control with permission checking.
"""

import sys
sys.path.append('..')

from authorization.rbac_manager import RBACManager, Permission


def main():
    print("=" * 70)
    print("  RBAC IMPLEMENTATION EXAMPLE")
    print("=" * 70 + "\n")
    
    # Initialize RBAC manager
    rbac = RBACManager()
    
    print("Phase 1: Role Definition")
    print("-" * 70)
    
    # Default roles already created (admin, user, analyst, editor)
    print("✓ Default roles created:")
    for role_name in ['admin', 'user', 'analyst', 'editor']:
        role_info = rbac.get_role_info(role_name)
        print(f"  - {role_name}: {len(role_info['permissions'])} permissions")
    
    print("\nPhase 2: User Assignment")
    print("-" * 70)
    
    # Simulate users
    users = {
        'alice': 'user_alice_001',
        'bob': 'user_bob_002',
        'charlie': 'user_charlie_003'
    }
    
    # Assign roles
    rbac.assign_role(users['alice'], 'admin')
    rbac.assign_role(users['bob'], 'analyst')
    rbac.assign_role(users['charlie'], 'user')
    
    print("✓ Users assigned to roles:")
    for name, user_id in users.items():
        roles = rbac.get_user_roles(user_id)
        print(f"  - {name}: {roles}")
    
    print("\nPhase 3: Permission Checking")
    print("-" * 70)
    
    # Check various permissions
    test_permissions = [
        Permission.READ_DOCUMENT,
        Permission.WRITE_DOCUMENT,
        Permission.MANAGE_USERS,
        Permission.EXPORT_DATA
    ]
    
    for name, user_id in users.items():
        print(f"\n{name}'s permissions:")
        for perm in test_permissions:
            has_perm = rbac.check_permission(user_id, perm)
            status = "✓" if has_perm else "✗"
            print(f"  {status} {perm.value}")
    
    print("\n\nPhase 4: Resource Protection")
    print("-" * 70)
    
    # Register protected resources
    rbac.register_resource(
        resource_id="doc_public",
        resource_type="document",
        owner_id=users['alice'],
        permissions_required=[Permission.READ_DOCUMENT]
    )
    
    rbac.register_resource(
        resource_id="doc_confidential",
        resource_type="document",
        owner_id=users['alice'],
        permissions_required=[Permission.READ_DOCUMENT, Permission.EXPORT_DATA]
    )
    
    print("✓ Resources registered with access requirements\n")
    
    # Check resource access
    for name, user_id in users.items():
        print(f"{name}'s resource access:")
        
        for resource_id in ["doc_public", "doc_confidential"]:
            access = rbac.check_resource_access(user_id, resource_id)
            status = "✓ Allowed" if access['allowed'] else "✗ Denied"
            print(f"  {status}: {resource_id}")
            if not access['allowed']:
                print(f"    Reason: {access['reason']}")
    
    print("\n" + "=" * 70)
    print("  RBAC DEMONSTRATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()