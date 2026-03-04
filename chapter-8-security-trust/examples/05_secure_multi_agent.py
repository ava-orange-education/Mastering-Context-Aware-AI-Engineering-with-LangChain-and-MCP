"""
Example 5: Secure Multi-Agent System
Demonstrates agent authentication, trust, and secure communication.
"""

import sys
sys.path.append('..')

from multi_agent_security.agent_authentication import AgentAuthenticator
from multi_agent_security.trust_manager import TrustManager
from multi_agent_security.message_signing import MessageSigner
from multi_agent_security.communication_guard import CommunicationGuard


def main():
    print("=" * 70)
    print("  SECURE MULTI-AGENT SYSTEM EXAMPLE")
    print("=" * 70 + "\n")
    
    # Initialize components
    agent_auth = AgentAuthenticator()
    trust_mgr = TrustManager()
    comm_guard = CommunicationGuard()
    
    print("Phase 1: Agent Registration")
    print("-" * 70)
    
    # Register agents
    agents = {}
    
    agent1 = agent_auth.register_agent(
        agent_id="research_agent",
        agent_name="Research Agent",
        permissions=["search", "read_documents"]
    )
    agents['research'] = agent1
    print(f"✓ Registered: Research Agent")
    
    agent2 = agent_auth.register_agent(
        agent_id="analysis_agent",
        agent_name="Analysis Agent",
        permissions=["analyze", "write_reports"]
    )
    agents['analysis'] = agent2
    print(f"✓ Registered: Analysis Agent")
    
    agent3 = agent_auth.register_agent(
        agent_id="execution_agent",
        agent_name="Execution Agent",
        permissions=["execute", "modify_data"]
    )
    agents['execution'] = agent3
    print(f"✓ Registered: Execution Agent")
    
    print("\n\nPhase 2: Agent Authentication")
    print("-" * 70)
    
    # Authenticate agents
    tokens = {}
    
    for name, agent_info in agents.items():
        auth_result = agent_auth.authenticate_agent(
            agent_id=agent_info['agent_id'],
            secret_key=agent_info['secret_key']
        )
        
        if auth_result['success']:
            tokens[name] = auth_result['token']
            print(f"✓ {auth_result['agent_name']} authenticated")
            print(f"  Token: {auth_result['token'][:20]}...")
            print(f"  Expires: {auth_result['expires_at']}")
    
    print("\n\nPhase 3: Token Verification")
    print("-" * 70)
    
    for name, token in tokens.items():
        verification = agent_auth.verify_token(token)
        
        if verification['valid']:
            print(f"✓ {verification['agent_name']}: Token valid")
            print(f"  Permissions: {verification['permissions']}")
        else:
            print(f"✗ Token invalid: {verification['error']}")
    
    print("\n\nPhase 4: Trust Relationships")
    print("-" * 70)
    
    # Establish trust between agents
    trust_mgr.establish_trust("research_agent", "analysis_agent", trust_level=1.0)
    trust_mgr.establish_trust("analysis_agent", "execution_agent", trust_level=0.8)
    trust_mgr.establish_trust("research_agent", "execution_agent", trust_level=0.6)
    
    print("✓ Trust relationships established\n")
    
    # Check trust levels
    for agent_from in ["research_agent", "analysis_agent"]:
        print(f"{agent_from} trusts:")
        trusted = trust_mgr.get_trusted_agents(agent_from, min_trust=0.5)
        for trusted_agent in trusted:
            level = trust_mgr.get_trust_level(agent_from, trusted_agent)
            print(f"  - {trusted_agent}: {level:.1f}")
    
    print("\n\nPhase 5: Secure Communication")
    print("-" * 70)
    
    # Establish shared key
    comm_guard.establish_shared_key("research_agent", "analysis_agent")
    print("✓ Shared encryption key established\n")
    
    # Send secure message
    message = "Research data: Customer satisfaction increased 15% in Q4"
    
    secure_package = comm_guard.secure_send(
        message=message,
        sender="research_agent",
        sender_key=agents['research']['secret_key'],
        recipient="analysis_agent"
    )
    
    print(f"Message sent (encrypted + signed):")
    print(f"  Original: {message}")
    print(f"  Encrypted: {secure_package['encrypted_message'][:40]}...")
    print(f"  Signature: {secure_package['signature'][:40]}...")
    
    # Receive and verify
    print(f"\nReceiving secure message...")
    
    received = comm_guard.secure_receive(
        package=secure_package,
        sender_key=agents['research']['secret_key']
    )
    
    print(f"✓ Message decrypted and verified:")
    print(f"  From: {received['sender']}")
    print(f"  Content: {received['message']}")
    print(f"  Verified: {received['verified']}")
    
    print("\n\nPhase 6: Security Scenarios")
    print("-" * 70)
    
    # Scenario 1: Expired token
    print("\nScenario 1: Revoke agent token")
    agent_auth.revoke_token(tokens['execution'])
    verification = agent_auth.verify_token(tokens['execution'])
    print(f"  Token valid after revocation: {verification['valid']}")
    
    # Scenario 2: Trust check before communication
    print("\nScenario 2: Trust-based access control")
    can_communicate = trust_mgr.is_trusted(
        "research_agent", 
        "execution_agent", 
        min_trust=0.7
    )
    print(f"  Research can send to Execution (min 0.7): {can_communicate}")
    
    can_communicate_lower = trust_mgr.is_trusted(
        "research_agent", 
        "execution_agent", 
        min_trust=0.5
    )
    print(f"  Research can send to Execution (min 0.5): {can_communicate_lower}")
    
    print("\n" + "=" * 70)
    print("  SECURE MULTI-AGENT DEMONSTRATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()