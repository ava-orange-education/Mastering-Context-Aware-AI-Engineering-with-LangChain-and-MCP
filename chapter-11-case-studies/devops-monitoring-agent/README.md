# DevOps Monitoring Agent

Autonomous AI agent for DevOps monitoring, incident detection, and automated response.

## Overview

This case study demonstrates a production-ready DevOps monitoring system that:

- Monitors infrastructure and application metrics
- Detects anomalies and incidents automatically
- Performs root cause analysis
- Executes automated remediation actions
- Integrates with major DevOps tools (Kubernetes, Prometheus, Datadog, PagerDuty)
- Learns from historical incidents

## Architecture
```
┌─────────────────────────────────────────────────┐
│           Metrics & Alerts Input                 │
│  (Prometheus, Datadog, CloudWatch, Logs)        │
└────────────────────┬────────────────────────────┘
                     │
         ┌───────────▼───────────┐
         │  Incident Detection   │
         │  (Anomaly Detection)  │
         └───────────┬───────────┘
                     │
        ┌────────────┼────────────┐
        │            │            │
        ▼            ▼            ▼
   ┌────────┐  ┌─────────┐  ┌──────────┐
   │Analysis│  │Root     │  │Action    │
   │Agent   │  │Cause    │  │Execution │
   │        │  │Agent    │  │Agent     │
   └────┬───┘  └────┬────┘  └────┬─────┘
        │           │            │
        └───────────┼────────────┘
                    │
           ┌────────┴────────┐
           │                 │
           ▼                 ▼
    ┌──────────┐      ┌──────────┐
    │Historical│      │Execution │
    │Incidents │      │Actions   │
    │RAG       │      │(K8s, AWS)│
    └──────────┘      └──────────┘
```

## Key Features

### Intelligent Monitoring
- Multi-source metric collection
- Anomaly detection with ML
- Pattern recognition
- Threshold-based and statistical alerting

### Automated Incident Response
- Automatic incident triage and prioritization
- Context-aware root cause analysis
- Intelligent remediation suggestions
- Safe automated action execution

### Learning System
- Historical incident analysis
- Runbook generation from past incidents
- Continuous improvement from outcomes
- Knowledge base of solutions

### Integration Ecosystem
- **Monitoring**: Prometheus, Datadog, New Relic, CloudWatch
- **Orchestration**: Kubernetes, Docker, AWS ECS
- **Incident Management**: PagerDuty, Opsgenie, ServiceNow
- **Communication**: Slack, MS Teams, Email
- **Cloud Providers**: AWS, GCP, Azure

## Quick Start

### Prerequisites

- Python 3.9+
- Anthropic API key
- Access to monitoring systems (Prometheus, Datadog, etc.)
- Kubernetes cluster (optional)
- AWS credentials (optional)

### Installation
```bash
cd devops-monitoring-agent

# Install dependencies
pip install -r ../requirements.txt

# Configure environment
cp ../.env.example ../.env
# Add your API keys and credentials
```

### Configuration

Required environment variables:
```bash
# Core
ANTHROPIC_API_KEY=your_key
VECTOR_STORE=pinecone

# Monitoring
PROMETHEUS_URL=http://prometheus:9090
DATADOG_API_KEY=your_datadog_key
DATADOG_APP_KEY=your_datadog_app_key

# Kubernetes
KUBECONFIG=/path/to/kubeconfig

# AWS
AWS_ACCESS_KEY_ID=your_key
AWS_SECRET_ACCESS_KEY=your_secret
AWS_REGION=us-east-1

# Incident Management
PAGERDUTY_API_KEY=your_key
SLACK_WEBHOOK_URL=your_webhook
```

### Running Examples

**Incident Detection:**
```bash
python examples/01_incident_detection.py
```

**Automated Response:**
```bash
python examples/02_automated_response.py
```

**Root Cause Analysis:**
```bash
python examples/03_root_cause_analysis.py
```

## Usage

### Basic Incident Detection
```python
from agents.incident_detection_agent import IncidentDetectionAgent
from monitoring.metrics_collector import MetricsCollector

# Initialize
detector = IncidentDetectionAgent()
collector = MetricsCollector()

# Collect metrics
metrics = await collector.collect_all_metrics()

# Detect incidents
result = await detector.process({
    "metrics": metrics,
    "timeframe": "5m"
})

if result.metadata.get("incidents_detected"):
    for incident in result.metadata["incidents"]:
        print(f"Incident: {incident['title']}")
        print(f"Severity: {incident['severity']}")
```

### Automated Remediation
```python
from agents.remediation_agent import RemediationAgent

remediation_agent = RemediationAgent()

# Analyze incident and suggest actions
result = await remediation_agent.process({
    "incident": incident_data,
    "system_state": current_state,
    "allow_automatic": True
})

# Execute safe actions
for action in result.metadata.get("actions", []):
    if action["safe_to_automate"]:
        await execute_action(action)
```

### Root Cause Analysis
```python
from agents.root_cause_agent import RootCauseAnalysisAgent

rca_agent = RootCauseAnalysisAgent()

# Perform root cause analysis
result = await rca_agent.process({
    "incident": incident_data,
    "metrics": related_metrics,
    "logs": relevant_logs,
    "recent_changes": deployments
})

print(f"Root Cause: {result.metadata['root_cause']}")
print(f"Contributing Factors: {result.metadata['factors']}")
```

## API Endpoints

Start the API server:
```bash
uvicorn api.main:app --reload --port 8003
```

### Endpoints

**Report Incident:**
```bash
POST /api/v1/incident
Content-Type: application/json

{
  "title": "High CPU usage on prod-api-1",
  "severity": "high",
  "metrics": {...},
  "context": {...}
}
```

**Get System Status:**
```bash
GET /api/v1/status
```

**Execute Remediation:**
```bash
POST /api/v1/remediate/{incident_id}
Content-Type: application/json

{
  "actions": ["restart_pod", "scale_up"],
  "dry_run": false
}
```

**Root Cause Analysis:**
```bash
POST /api/v1/analyze/{incident_id}
```

## Monitoring Integrations

### Prometheus
```python
from integrations.prometheus_connector import PrometheusConnector

prometheus = PrometheusConnector(url="http://prometheus:9090")

# Query metrics
cpu_usage = await prometheus.query(
    "rate(container_cpu_usage_seconds_total[5m])"
)

# Get alerts
alerts = await prometheus.get_active_alerts()
```

### Datadog
```python
from integrations.datadog_connector import DatadogConnector

datadog = DatadogConnector()

# Get monitors
monitors = await datadog.get_monitors(state="alert")

# Query metrics
metrics = await datadog.query_metrics(
    query="avg:system.cpu.user{*}",
    start=start_time,
    end=end_time
)
```

### Kubernetes
```python
from integrations.kubernetes_connector import KubernetesConnector

k8s = KubernetesConnector()

# Get pod status
pods = await k8s.get_pods(namespace="production")

# Scale deployment
await k8s.scale_deployment(
    name="api-server",
    namespace="production",
    replicas=5
)
```

## Automated Actions

### Safe Actions (Auto-Executable)

- Restart unhealthy pods
- Scale deployments up/down
- Clear cache
- Reload configuration
- Trigger garbage collection

### Supervised Actions (Require Approval)

- Database migrations
- Traffic routing changes
- Instance termination
- Major configuration changes
- Resource deletion

### Action Safety Framework
```python
class ActionSafety:
    """
    Safety framework for automated actions
    """
    
    SAFE_ACTIONS = [
        "restart_pod",
        "scale_deployment",
        "clear_cache"
    ]
    
    SUPERVISED_ACTIONS = [
        "terminate_instance",
        "modify_security_group",
        "change_routing"
    ]
    
    def is_safe_to_automate(action: str) -> bool:
        return action in SAFE_ACTIONS
```

## Incident Response Workflow

1. **Detection**: Monitor metrics and detect anomalies
2. **Classification**: Categorize incident by type and severity
3. **Analysis**: Gather context and perform root cause analysis
4. **Action Planning**: Generate remediation plan
5. **Execution**: Execute safe automated actions
6. **Escalation**: Escalate if automation insufficient
7. **Learning**: Update knowledge base with resolution

## Root Cause Analysis

### Analysis Techniques

- **Correlation Analysis**: Find correlated metric changes
- **Timeline Analysis**: Reconstruct event sequence
- **Change Analysis**: Identify recent deployments/changes
- **Pattern Matching**: Compare with historical incidents
- **Dependency Analysis**: Trace through system dependencies

### RCA Output
```json
{
  "root_cause": "Memory leak in payment-service v2.1.3",
  "confidence": 0.85,
  "evidence": [
    "Memory usage increased linearly over 6 hours",
    "Deployment of v2.1.3 at incident start time",
    "Similar incident with v2.1.2 two weeks ago"
  ],
  "contributing_factors": [
    "Insufficient memory limits",
    "Missing memory monitoring alerts"
  ],
  "recommended_actions": [
    "Rollback to v2.1.2",
    "Add memory profiling to CI/CD",
    "Set memory limits to 2GB"
  ]
}
```

## Runbook Generation

The system automatically generates runbooks from resolved incidents:
```yaml
runbook:
  title: "High Memory Usage - Payment Service"
  triggers:
    - metric: memory_usage_percent
      threshold: 85
      service: payment-service
  
  symptoms:
    - Slow API responses
    - Increasing pod restarts
    - OOM errors in logs
  
  diagnosis_steps:
    - Check memory metrics for service
    - Review recent deployments
    - Examine application logs for memory errors
  
  remediation_steps:
    - Restart affected pods
    - Scale up if persistent
    - Rollback recent deployment if correlated
    - Add memory limits if missing
  
  prevention:
    - Add memory usage alerts at 70%
    - Implement memory profiling in tests
    - Set appropriate resource limits
```

## Evaluation

### Detection Accuracy
```python
from evaluation.detection_accuracy import DetectionAccuracyEvaluator

evaluator = DetectionAccuracyEvaluator()

metrics = evaluator.evaluate(
    true_incidents=labeled_incidents,
    detected_incidents=system_detections
)

print(f"Precision: {metrics['precision']:.2%}")
print(f"Recall: {metrics['recall']:.2%}")
print(f"False Positive Rate: {metrics['false_positive_rate']:.2%}")
```

### Action Effectiveness
```python
from evaluation.action_effectiveness import ActionEffectivenessEvaluator

evaluator = ActionEffectivenessEvaluator()

effectiveness = evaluator.evaluate_action(
    action="restart_pod",
    incident=incident_data,
    outcome=resolution_data
)

print(f"Success Rate: {effectiveness['success_rate']:.2%}")
print(f"Mean Time to Resolution: {effectiveness['mttr']:.1f}s")
```

## Metrics & KPIs

- **MTTD** (Mean Time to Detect): How quickly incidents are detected
- **MTTR** (Mean Time to Resolve): How quickly incidents are resolved
- **Automation Rate**: Percentage of incidents resolved automatically
- **False Positive Rate**: Incorrect incident detections
- **Action Success Rate**: Successful automated remediation

## Testing
```bash
# Run all tests
pytest tests/

# Test specific components
pytest tests/test_agents.py
pytest tests/test_integrations.py
pytest tests/test_actions.py
```

## Production Deployment

See `docs/deployment_guide.md` for:
- Kubernetes deployment manifests
- High availability configuration
- Security best practices
- Scaling recommendations
- Monitoring the monitor

## Safety & Governance

### Safety Mechanisms

- **Action Validation**: Verify actions before execution
- **Dry Run Mode**: Test actions without execution
- **Rollback Capability**: Undo automated changes
- **Rate Limiting**: Prevent action storms
- **Circuit Breakers**: Stop automation if failure rate high

### Governance

- **Action Auditing**: Log all automated actions
- **Approval Workflows**: Human approval for risky actions
- **Impact Assessment**: Estimate blast radius
- **Compliance Checks**: Ensure regulatory compliance

## Limitations

- Requires good metric coverage
- Historical data needed for ML models
- Cloud-specific integrations may vary
- Complex distributed systems are challenging
- Novel issues may not be handled well

## Future Enhancements

- Multi-cloud support
- Advanced ML anomaly detection
- Natural language incident reporting
- Predictive incident prevention
- Integration with more tools

## Support

- **Documentation**: See `docs/` directory
- **Examples**: See `examples/` directory
- **Issues**: GitHub Issues

## License

Enterprise software - see LICENSE for details.