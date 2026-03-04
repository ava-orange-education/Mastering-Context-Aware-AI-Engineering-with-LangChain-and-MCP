"""
Example 9: Complete secure AI system with all security features.
"""

import sys
sys.path.append('..')

from authentication.auth_manager import AuthenticationManager
from authorization.rbac_manager import RBACManager, Permission
from security.hallucination_detector import HallucinationDetector
from data_protection.pii_detector import PIIDetector
from data_protection.data_anonymizer import DataAnonymizer
from grounding.citation_manager import CitationManager
from secure_rag.secure_retriever import SecureRetriever
from multi_agent_security.agent_authentication import AgentAuthenticator
from monitoring.security_monitor import SecurityMonitor
from anthropic import Anthropic
import os


def main():
    print("=" * 70)
    print("  COMPLETE SECURE AI SYSTEM DEMONSTRATION")
    print("=" * 70 + "\n")
    
    # ===== INITIALIZE SECURITY COMPONENTS =====
    print("Phase 1: Initializing Security Components")
    print("-" * 70)
    
    # Authentication
    auth_manager = AuthenticationManager(session_timeout_minutes=30)
    
    # Authorization (RBAC)
    rbac_manager = RBACManager()
    
    # Security Monitoring
    security_monitor = SecurityMonitor()
    
    # Content Security
    llm_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    hallucination_detector = HallucinationDetector(llm_client)
    pii_detector = PIIDetector()
    data_anonymizer = DataAnonymizer(pii_detector)
    
    # Grounding
    citation_manager = CitationManager()
    
    # Agent Security
    agent_auth = AgentAuthenticator()
    
    print("✓ All security components initialized\n")
    
    # ===== USER REGISTRATION AND AUTHENTICATION =====
    print("Phase 2: User Management")
    print("-" * 70)
    
    # Register users
    user1_reg = auth_manager.register_user(
        username="alice_admin",
        email="alice@example.com",
        password="SecurePass123!",
        roles=["admin"]
    )
    
    user2_reg = auth_manager.register_user(
        username="bob_analyst",
        email="bob@example.com",
        password="SecurePass456!",
        roles=["analyst"]
    )
    
    user3_reg = auth_manager.register_user(
        username="charlie_user",
        email="charlie@example.com",
        password="SecurePass789!",
        roles=["user"]
    )
    
    print(f"✓ Registered users: alice_admin, bob_analyst, charlie_user")
    
    # Assign roles
    rbac_manager.assign_role(user1_reg['user_id'], "admin")
    rbac_manager.assign_role(user2_reg['user_id'], "analyst")
    rbac_manager.assign_role(user3_reg['user_id'], "user")
    
    print("✓ Roles assigned\n")
    
    # ===== AUTHENTICATION =====
    print("Phase 3: Authentication")
    print("-" * 70)
    
    # Authenticate Alice (admin)
    alice_auth = auth_manager.authenticate(
        username="alice_admin",
        password="SecurePass123!",
        ip_address="192.168.1.10"
    )
    
    if alice_auth['success']:
        print(f"✓ Alice authenticated (admin)")
        print(f"  Session: {alice_auth['session_token'][:20]}...")
        alice_session = alice_auth['session_token']
        alice_id = alice_auth['user_id']
    
    # Test failed authentication
    failed_auth = auth_manager.authenticate(
        username="bob_analyst",
        password="WrongPassword",
        ip_address="192.168.1.20"
    )
    
    print(f"✗ Bob's failed auth: {failed_auth['error']}")
    security_monitor.log_failed_authentication("bob_analyst", "192.168.1.20")
    
    # Successful authentication for Bob
    bob_auth = auth_manager.authenticate(
        username="bob_analyst",
        password="SecurePass456!",
        ip_address="192.168.1.20"
    )
    
    if bob_auth['success']:
        print(f"✓ Bob authenticated (analyst)")
        bob_id = bob_auth['user_id']
    
    print()
    
    # ===== AUTHORIZATION AND RBAC =====
    print("Phase 4: Authorization Testing")
    print("-" * 70)
    
    # Check permissions
    alice_can_manage = rbac_manager.check_permission(alice_id, Permission.MANAGE_USERS)
    bob_can_manage = rbac_manager.check_permission(bob_id, Permission.MANAGE_USERS)
    bob_can_export = rbac_manager.check_permission(bob_id, Permission.EXPORT_DATA)
    
    print(f"Alice (admin) can manage users: {alice_can_manage}")
    print(f"Bob (analyst) can manage users: {bob_can_manage}")
    print(f"Bob (analyst) can export data: {bob_can_export}")
    
    # Register protected resources
    rbac_manager.register_resource(
        resource_id="doc_001",
        resource_type="document",
        owner_id=alice_id,
        permissions_required=[Permission.READ_DOCUMENT],
        metadata={'title': 'Public Document', 'sensitivity': 'public'}
    )
    
    rbac_manager.register_resource(
        resource_id="doc_002",
        resource_type="document",
        owner_id=alice_id,
        permissions_required=[Permission.READ_DOCUMENT, Permission.EXPORT_DATA],
        metadata={'title': 'Sensitive Report', 'sensitivity': 'confidential'}
    )
    
    # Check resource access
    bob_access_doc1 = rbac_manager.check_resource_access(bob_id, "doc_001")
    bob_access_doc2 = rbac_manager.check_resource_access(bob_id, "doc_002")
    
    print(f"\nBob can access doc_001: {bob_access_doc1['allowed']}")
    print(f"Bob can access doc_002: {bob_access_doc2['allowed']}")
    
    if not bob_access_doc2['allowed']:
        security_monitor.log_unauthorized_access(
            bob_id, "doc_002",
            ['READ_DOCUMENT', 'EXPORT_DATA']
        )
    
    print()
    
    # ===== PII DETECTION AND ANONYMIZATION =====
    print("Phase 5: PII Protection")
    print("-" * 70)
    
    test_text = """
    Contact information:
    - Email: john.doe@company.com
    - Phone: 555-123-4567
    - SSN: 123-45-6789
    - Address: 123 Main Street, Springfield
    """
    
    # Detect PII
    pii_scan = pii_detector.scan_document(test_text)
    
    print(f"PII detected: {pii_scan['contains_pii']}")
    print(f"PII count: {pii_scan['pii_count']}")
    print(f"PII types found: {list(pii_scan['summary'].keys())}")
    
    # Anonymize
    anonymized = data_anonymizer.anonymize(test_text, strategy="mask")
    
    print(f"\nOriginal text excerpt: {test_text[:100]}...")
    print(f"Anonymized text excerpt: {anonymized['anonymized_text'][:100]}...")
    print(f"Replacements made: {anonymized['replacements']}")
    
    # Log PII access
    security_monitor.log_pii_access(
        bob_id, "doc_001",
        pii_types=list(pii_scan['summary'].keys())
    )
    
    print()
    
    # ===== HALLUCINATION DETECTION =====
    print("Phase 6: Hallucination Detection")
    print("-" * 70)
    
    # Test response with potential hallucination
    test_response = """
    Studies show that 95% of all companies will adopt AI by 2025.
    Research indicates that the AI market will definitely reach $500 billion.
    Experts say this is absolutely guaranteed to happen.
    """
    
    hallucination_check = hallucination_detector.check_response(test_response)
    
    print(f"Hallucination detected: {hallucination_check.is_hallucination}")
    print(f"Confidence: {hallucination_check.confidence:.2f}")
    print(f"Reason: {hallucination_check.reason}")
    print(f"Severity: {hallucination_check.severity}")
    
    if hallucination_check.is_hallucination:
        security_monitor.log_hallucination_detected(
            response_id="resp_001",
            confidence=hallucination_check.confidence,
            reason=hallucination_check.reason
        )
    
    print()
    
    # ===== CITATION AND GROUNDING =====
    print("Phase 7: Citation Management")
    print("-" * 70)
    
    # Create cited response
    response_text = "AI adoption is accelerating across industries, with significant growth in healthcare and finance sectors."
    
    source_docs = [
        {
            'title': 'AI Trends Report 2024',
            'url': 'https://example.com/ai-trends',
            'excerpt': 'AI adoption accelerating with 40% YoY growth',
            'type': 'report'
        },
        {
            'title': 'Healthcare AI Applications',
            'url': 'https://example.com/healthcare-ai',
            'excerpt': 'Healthcare sector leading AI implementation',
            'type': 'article'
        }
    ]
    
    cited_response = citation_manager.create_cited_response(
        response_text, source_docs
    )
    
    formatted_response = citation_manager.format_response_with_citations(cited_response)
    
    print("Cited Response:")
    print(formatted_response)
    print(f"\nGrounding score: {cited_response.grounding_score:.2f}")
    
    # Verify citations
    verification = citation_manager.verify_citations(
        response_text, cited_response.citations
    )
    
    print(f"Well grounded: {verification['well_grounded']}")
    print(f"Grounding rate: {verification['grounding_rate']:.2f}")
    
    print()
    
    # ===== MULTI-AGENT SECURITY =====
    print("Phase 8: Multi-Agent Security")
    print("-" * 70)
    
    # Register agents
    agent1_reg = agent_auth.register_agent(
        agent_id="agent_research",
        agent_name="Research Agent",
        permissions=["search", "read_documents"]
    )
    
    agent2_reg = agent_auth.register_agent(
        agent_id="agent_analysis",
        agent_name="Analysis Agent",
        permissions=["read_documents", "write_reports"]
    )
    
    print(f"✓ Registered agents: Research Agent, Analysis Agent")
    
    # Authenticate agent
    agent1_auth = agent_auth.authenticate_agent(
        agent_id="agent_research",
        secret_key=agent1_reg['secret_key']
    )
    
    if agent1_auth['success']:
        print(f"✓ Research Agent authenticated")
        print(f"  Token: {agent1_auth['token'][:20]}...")
        agent1_token = agent1_auth['token']
    
    # Verify agent token
    token_verification = agent_auth.verify_token(agent1_token)
    
    print(f"✓ Token valid: {token_verification['valid']}")
    print(f"  Agent: {token_verification['agent_name']}")
    print(f"  Permissions: {token_verification['permissions']}")
    
    print()
    
    # ===== SECURITY MONITORING =====
    print("Phase 9: Security Monitoring Summary")
    print("-" * 70)
    
    # Get security summary
    security_summary = security_monitor.get_security_summary()
    
    print(f"Recent events (last hour): {security_summary['recent_events']}")
    print(f"Active alerts: {security_summary['active_alerts']}")
    print(f"Total events logged: {security_summary['total_events']}")
    
    print("\nEvent types:")
    for event_type, count in security_summary['event_types'].items():
        print(f"  - {event_type}: {count}")
    
    print("\nSeverity distribution:")
    for severity, count in security_summary['severity_distribution'].items():
        print(f"  - {severity}: {count}")
    
    # Get active alerts
    active_alerts = security_monitor.get_active_alerts()
    
    if active_alerts:
        print(f"\n⚠ ACTIVE SECURITY ALERTS: {len(active_alerts)}")
        for alert in active_alerts:
            print(f"  - [{alert.severity.upper()}] {alert.message}")
    
    print()
    
    # ===== SYSTEM STATUS =====
    print("=" * 70)
    print("  SECURE SYSTEM STATUS")
    print("=" * 70)
    
    print(f"\n✓ Authentication: {len(auth_manager.users)} users, {len(auth_manager.sessions)} active sessions")
    print(f"✓ Authorization: {len(rbac_manager.roles)} roles, {len(rbac_manager.resources)} protected resources")
    print(f"✓ Agents: {len(agent_auth.agents)} registered agents")
    print(f"✓ Citations: {len(citation_manager.citations)} citations tracked")
    print(f"✓ Security Events: {len(security_monitor.events)} logged")
    print(f"✓ Security Alerts: {len(active_alerts)} active, {len(security_monitor.alerts)} total")
    
    print("\n" + "=" * 70)
    print("  DEMONSTRATION COMPLETE")
    print("=" * 70)
    
    print("\nKey Security Features Demonstrated:")
    print("  ✓ User authentication with session management")
    print("  ✓ Role-Based Access Control (RBAC)")
    print("  ✓ PII detection and anonymization")
    print("  ✓ Hallucination detection")
    print("  ✓ Citation management and grounding")
    print("  ✓ Multi-agent authentication")
    print("  ✓ Real-time security monitoring")
    print("  ✓ Alert system for security threats")


if __name__ == "__main__":
    main()