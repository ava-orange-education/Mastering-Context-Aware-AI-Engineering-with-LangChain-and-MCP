# Observability Setup Guide

This guide walks through setting up the complete observability stack for AI agent monitoring.

## Overview

The observability stack includes:
- **Langfuse**: Trace visualization and analysis
- **LangSmith**: LLM monitoring and debugging  
- **Prometheus**: Metrics collection
- **Grafana**: Metrics visualization
- **Custom Trace Logger**: Detailed execution traces

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure API Keys

Create a `.env` file:
```bash
# Langfuse
LANGFUSE_PUBLIC_KEY=your_public_key
LANGFUSE_SECRET_KEY=your_secret_key

# LangSmith  
LANGCHAIN_API_KEY=your_api_key

# Anthropic
ANTHROPIC_API_KEY=your_api_key
```

### 3. Initialize Components
```python
from observability.langfuse_integration import LangfuseIntegration
from observability.prometheus_exporter import PrometheusExporter, PrometheusServer
from observability.trace_logger import TraceLogger

# Initialize
langfuse = LangfuseIntegration()
prometheus = PrometheusExporter()
trace_logger = TraceLogger()

# Start Prometheus server
server = PrometheusServer(prometheus, port=8000)
server.start()
```

## Langfuse Setup

### Sign Up

1. Go to https://langfuse.com
2. Create an account
3. Create a new project
4. Copy your API keys

### Configuration
```yaml
# configs/langfuse_config.yaml
langfuse:
  enabled: true
  public_key: ${LANGFUSE_PUBLIC_KEY}
  secret_key: ${LANGFUSE_SECRET_KEY}
  host: "https://cloud.langfuse.com"
```

### Usage
```python
# Trace agent run
trace_id = langfuse.trace_agent_run(
    agent_name="QuestionAnswering",
    query="What is AI?",
    response="AI is artificial intelligence...",
    metadata={'model': 'claude-3-5-sonnet'}
)

# Log evaluation scores
langfuse.log_score(trace_id, "relevance", 0.92)
```

## LangSmith Setup

### Sign Up

1. Go to https://smith.langchain.com
2. Create an account
3. Get your API key

### Configuration
```python
from observability.langsmith_integration import LangSmithIntegration

langsmith = LangSmithIntegration(api_key="your_api_key")

# Log agent run
langsmith.log_agent_run(
    agent_name="ResearchAgent",
    query="Latest AI developments",
    response="Recent developments include..."
)
```

## Prometheus Setup

### Installation
```bash
# macOS
brew install prometheus

# Linux
wget https://github.com/prometheus/prometheus/releases/download/v2.45.0/prometheus-2.45.0.linux-amd64.tar.gz
tar xvf prometheus-*.tar.gz
cd prometheus-*
```

### Configuration

Create `prometheus.yml`:
```yaml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'ai-agent'
    static_configs:
      - targets: ['localhost:8000']
```

### Start Prometheus
```bash
./prometheus --config.file=prometheus.yml
```

Access at: http://localhost:9090

## Grafana Setup

### Installation
```bash
# macOS
brew install grafana

# Linux
sudo apt-get install -y grafana
```

### Start Grafana
```bash
# macOS
brew services start grafana

# Linux
sudo systemctl start grafana-server
```

Access at: http://localhost:3000 (default credentials: admin/admin)

### Add Prometheus Data Source

1. Go to Configuration → Data Sources
2. Click "Add data source"
3. Select "Prometheus"
4. Set URL: http://localhost:9090
5. Click "Save & Test"

### Import Dashboards
```bash
# Import pre-built dashboards
grafana-cli dashboards import configs/grafana_dashboards.json
```

Or manually:
1. Go to Dashboards → Import
2. Upload `grafana_evaluation.json`
3. Select Prometheus as data source
4. Click Import

## Custom Trace Logger

### Setup
```python
from observability.trace_logger import TraceLogger

trace_logger = TraceLogger(log_file="traces.jsonl")

# Start trace
trace_id = trace_logger.start_trace("agent_name", "query")

# Log steps
trace_logger.log_step(trace_id, "thought", "Analyzing query...")
trace_logger.log_step(trace_id, "action", "Searching documents...")
trace_logger.log_step(trace_id, "observation", "Found 5 documents")

# End trace
trace_logger.end_trace(trace_id, response="Final response", status="completed")
```

### View Traces
```python
# Get recent traces
recent = trace_logger.get_recent_traces(count=10)

# Export traces
trace_logger.export_traces("traces_export.json")
```

## Monitoring Best Practices

### 1. Metrics to Track

**Performance:**
- Request latency (P50, P95, P99)
- Throughput (requests/second)
- Error rate

**Quality:**
- Evaluation scores (relevance, coherence, etc.)
- User feedback
- Hallucination rate

**Resources:**
- Token usage
- API costs
- System resources

### 2. Alerting

Set up alerts for:
- High error rate (>5%)
- High latency (P95 >5s)
- Low evaluation scores (<0.7)
- Cost spikes

### 3. Dashboards

Create dashboards for:
- Real-time performance metrics
- Evaluation score trends
- Error analysis
- Cost tracking

## Troubleshooting

### Langfuse Not Receiving Traces

1. Check API keys are correct
2. Verify network connectivity
3. Call `langfuse.flush()` to send pending traces
4. Enable debug mode: `debug_mode: true`

### Prometheus Not Scraping Metrics

1. Verify metrics server is running on correct port
2. Check `prometheus.yml` configuration
3. Test metrics endpoint: `curl http://localhost:8000/metrics`
4. Check Prometheus targets page

### Grafana Dashboard Empty

1. Verify Prometheus data source is connected
2. Check time range selector
3. Verify metrics exist in Prometheus
4. Check dashboard queries

## Examples

See `examples/08_complete_observability.py` for a complete working example.

## Next Steps

1. Set up custom alerts
2. Create team-specific dashboards
3. Integrate with incident management (PagerDuty, Opsgenie)
4. Set up log aggregation (ELK, Splunk)
5. Implement distributed tracing