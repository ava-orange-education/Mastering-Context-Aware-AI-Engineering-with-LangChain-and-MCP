"""
Example 2: Incident Response

Demonstrates end-to-end incident response workflow
"""

import asyncio
import sys
sys.path.append('..')

from agents.incident_detection_agent import IncidentDetectionAgent
from agents.root_cause_analysis_agent import RootCauseAnalysisAgent
from agents.remediation_agent import RemediationAgent
from datetime import datetime


async def main():
    """Run incident response example"""
    
    print("="*70)
    print("DevOps Monitoring - Incident Response Example")
    print("="*70)
    print()
    
    # Initialize agents
    print("Initializing agents...")
    detection_agent = IncidentDetectionAgent()
    rca_agent = RootCauseAnalysisAgent()
    remediation_agent = RemediationAgent()
    
    # Step 1: Incident Detection
    print("\n" + "="*70)
    print("STEP 1: INCIDENT DETECTION")
    print("="*70)
    
    incident_data = {
        "metrics": {
            "api_server_cpu": 0.95,
            "api_server_memory": 0.88,
            "api_server_error_rate": 0.12,
            "api_server_response_time": 2500,
            "database_connections": 450,
            "queue_depth": 5000
        },
        "alerts": [
            "High CPU usage on api-server-1",
            "High memory usage on api-server-1",
            "Error rate above threshold",
            "Response time SLA violation"
        ],
        "logs": [
            "ERROR: Database connection pool exhausted",
            "ERROR: Timeout waiting for database connection",
            "WARNING: High request queue depth",
            "ERROR: Out of memory exception in request handler"
        ]
    }
    
    print("\nIncoming Signals:")
    print(f"  Metrics: {len(incident_data['metrics'])} abnormal metrics")
    print(f"  Alerts: {len(incident_data['alerts'])} active alerts")
    print(f"  Logs: {len(incident_data['logs'])} error logs")
    
    print("\nDetecting incident...")
    detection_response = await detection_agent.process(incident_data)
    
    incident = detection_response.metadata
    
    print(f"\n✓ Incident Detected!")
    print(f"  Title: {incident['title']}")
    print(f"  Severity: {incident['severity'].upper()}")
    print(f"  Type: {incident['type']}")
    print(f"  Affected Components:")
    for component in incident['affected_components']:
        print(f"    - {component}")
    
    # Step 2: Root Cause Analysis
    print("\n" + "="*70)
    print("STEP 2: ROOT CAUSE ANALYSIS")
    print("="*70)
    
    rca_input = {
        "incident": incident,
        "metrics": incident_data["metrics"],
        "logs": incident_data["logs"],
        "recent_changes": [
            {
                "type": "deployment",
                "description": "Deployed api-server v2.5.0",
                "timestamp": "2024-01-15T14:30:00Z"
            },
            {
                "type": "config_change",
                "description": "Updated database connection pool size",
                "timestamp": "2024-01-15T13:00:00Z"
            }
        ],
        "system_state": {
            "api_server_replicas": 3,
            "database_max_connections": 500,
            "current_connections": 450
        }
    }
    
    print("\nPerforming root cause analysis...")
    rca_response = await rca_agent.process(rca_input)
    
    rca = rca_response.metadata
    
    print(f"\n✓ Root Cause Identified!")
    print(f"  Primary Cause: {rca['root_cause']}")
    
    if rca.get('contributing_factors'):
        print(f"\n  Contributing Factors:")
        for factor in rca['contributing_factors'][:3]:
            print(f"    - {factor}")
    
    if rca.get('prevention_measures'):
        print(f"\n  Prevention Measures:")
        for measure in rca['prevention_measures'][:3]:
            print(f"    - {measure}")
    
    print(f"\n  Confidence: {rca['confidence']:.0%}")
    
    # Step 3: Remediation
    print("\n" + "="*70)
    print("STEP 3: REMEDIATION")
    print("="*70)
    
    remediation_input = {
        "incident": incident,
        "root_cause": rca['root_cause'],
        "system_state": rca_input["system_state"],
        "allow_automatic": True,
        "constraints": {
            "max_replicas": 10,
            "maintenance_window": False
        }
    }
    
    print("\nGenerating remediation plan...")
    remediation_response = await remediation_agent.process(remediation_input)
    
    actions = remediation_response.metadata.get('actions', [])
    safe_actions = remediation_response.metadata.get('safe_actions', [])
    
    print(f"\n✓ Remediation Plan Generated!")
    print(f"  Total Actions: {len(actions)}")
    print(f"  Safe to Auto-execute: {len(safe_actions)}")
    
    print("\n  Recommended Actions:")
    for i, action in enumerate(actions[:3], 1):
        print(f"\n  {i}. {action['action']}")
        print(f"     Description: {action['description']}")
        print(f"     Safe to automate: {'Yes' if action['safe_to_automate'] else 'No'}")
        print(f"     Impact: {action.get('impact', 'Unknown')}")
    
    # Step 4: Execution Summary
    print("\n" + "="*70)
    print("STEP 4: EXECUTION SUMMARY")
    print("="*70)
    
    print("\nIncident Response Timeline:")
    print(f"  {datetime.utcnow().strftime('%H:%M:%S')} - Incident detected")
    print(f"  {datetime.utcnow().strftime('%H:%M:%S')} - Root cause identified")
    print(f"  {datetime.utcnow().strftime('%H:%M:%S')} - Remediation plan created")
    print(f"  {datetime.utcnow().strftime('%H:%M:%S')} - Safe actions ready for execution")
    
    print("\nNext Steps:")
    print("  1. Execute safe actions automatically")
    print("  2. Request approval for supervised actions")
    print("  3. Monitor incident resolution")
    print("  4. Update runbooks based on learnings")
    print("  5. Schedule post-incident review")
    
    print("\nMetrics to Monitor:")
    print("  • CPU usage trend")
    print("  • Memory usage trend")
    print("  • Error rate")
    print("  • Response time")
    print("  • Database connection pool utilization")
    
    print("\n" + "="*70)
    print("Example completed!")
    print("="*70)


if __name__ == "__main__":
    asyncio.run(main())