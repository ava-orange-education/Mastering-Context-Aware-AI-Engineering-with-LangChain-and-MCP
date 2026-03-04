"""
Example 7: PII Protection Pipeline
Demonstrates comprehensive PII protection workflow.
"""

import sys
sys.path.append('..')

from data_protection.pii_detector import PIIDetector
from data_protection.data_anonymizer import DataAnonymizer
from authorization.rbac_manager import RBACManager, Permission
from monitoring.security_monitor import SecurityMonitor


def main():
    print("=" * 70)
    print("  PII PROTECTION PIPELINE EXAMPLE")
    print("=" * 70 + "\n")
    
    # Initialize components
    pii_detector = PIIDetector()
    anonymizer = DataAnonymizer(pii_detector)
    rbac = RBACManager()
    security_monitor = SecurityMonitor()
    
    print("Phase 1: Setup Users with Different Permissions")
    print("-" * 70)
    
    # Create users
    alice_id = "user_alice_001"  # Admin - can view PII
    bob_id = "user_bob_002"      # Analyst - can export (view PII)
    charlie_id = "user_charlie_003"  # User - cannot view PII
    
    rbac.assign_role(alice_id, "admin")
    rbac.assign_role(bob_id, "analyst")
    rbac.assign_role(charlie_id, "user")
    
    print(f"✓ Alice (admin): {rbac.get_user_permissions(alice_id)[:3]}")
    print(f"✓ Bob (analyst): {rbac.get_user_permissions(bob_id)[:3]}")
    print(f"✓ Charlie (user): {rbac.get_user_permissions(charlie_id)[:3]}")
    
    print("\n\nPhase 2: Document with PII")
    print("-" * 70)
    
    document = """
    Customer Support Ticket #12345
    
    Customer: Jane Doe
    Email: jane.doe@email.com
    Phone: 555-123-4567
    Account: 4532-****-****-9876
    
    Issue: Customer reported unauthorized charge of $49.99 on 03/15/2024.
    Requested refund to be processed to card ending in 9876.
    
    Agent notes: Verified customer identity via last 4 of SSN: 6789
    Approved refund. Case closed.
    """
    
    print(f"Original document:\n{document}\n")
    
    # Detect PII
    pii_scan = pii_detector.scan_document(document)
    
    print(f"PII Analysis:")
    print(f"  Contains PII: {pii_scan['contains_pii']}")
    print(f"  PII count: {pii_scan['pii_count']}")
    print(f"  Types: {list(pii_scan['summary'].keys())}")
    print(f"  High-risk PII: {pii_detector.has_high_risk_pii(document)}")
    
    print("\n\nPhase 3: Role-Based PII Filtering")
    print("-" * 70)
    
    users_to_test = [
        ('Alice (admin)', alice_id),
        ('Bob (analyst)', bob_id),
        ('Charlie (user)', charlie_id)
    ]
    
    for name, user_id in users_to_test:
        print(f"\n{name}:")
        
        # Check if user can view PII
        can_view_pii = rbac.check_permission(user_id, Permission.EXPORT_DATA)
        print(f"  Can view PII: {can_view_pii}")
        
        if can_view_pii:
            # User sees full document
            filtered_doc = document
            print(f"  Access: Full document (no filtering)")
            
            # Log PII access
            security_monitor.log_pii_access(
                user_id=user_id,
                document_id="ticket_12345",
                pii_types=list(pii_scan['summary'].keys())
            )
        else:
            # Anonymize PII
            result = anonymizer.anonymize(document, strategy='mask')
            filtered_doc = result['anonymized_text']
            print(f"  Access: Anonymized ({result['replacements']} PII instances masked)")
        
        print(f"  Preview: {filtered_doc[:100]}...")
    
    print("\n\nPhase 4: Anonymization Strategies Comparison")
    print("-" * 70)
    
    strategies = {
        'redact': 'Complete removal',
        'mask': 'Partial masking',
        'pseudonymize': 'Fake but consistent values',
        'hash': 'One-way hash'
    }
    
    for strategy, description in strategies.items():
        print(f"\n{strategy.upper()} ({description}):")
        result = anonymizer.anonymize(document, strategy=strategy)
        
        # Show sample of anonymized content
        lines = result['anonymized_text'].split('\n')
        for line in lines[3:6]:  # Show a few lines
            if line.strip():
                print(f"  {line}")
    
    print("\n\nPhase 5: Security Monitoring")
    print("-" * 70)
    
    # Check security events
    recent_events = security_monitor.get_recent_events(minutes=60)
    pii_events = [e for e in recent_events if e.event_type == 'pii_access']
    
    print(f"\nPII Access Events: {len(pii_events)}")
    for event in pii_events:
        print(f"  - User {event.user_id}: {len(event.metadata.get('pii_types', []))} PII types")
    
    # Check for excessive PII access
    from collections import defaultdict
    access_counts = defaultdict(int)
    for event in pii_events:
        if event.user_id:
            access_counts[event.user_id] += 1
    
    print(f"\nAccess frequency:")
    for user_id, count in access_counts.items():
        print(f"  {user_id}: {count} accesses")
        if count > 10:
            print(f"    ⚠ Warning: Excessive PII access detected")
    
    print("\n\nPhase 6: Compliance Report")
    print("-" * 70)
    
    print(f"\nPII Protection Summary:")
    print(f"  ✓ PII detection: Active")
    print(f"  ✓ Role-based filtering: Enforced")
    print(f"  ✓ Access logging: Complete")
    print(f"  ✓ Anonymization: Available")
    print(f"  ✓ Security monitoring: Active")
    
    print("\n" + "=" * 70)
    print("  PII PROTECTION DEMONSTRATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()