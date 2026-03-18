# Chapter 9: Observability and Evaluation of AI Agents

Complete implementation of observability and evaluation framework for AI agent systems.

## Features

### Evaluation Metrics
- **Groundedness**: Measure response grounding in source material
- **Coherence**: Evaluate logical flow and consistency
- **Factuality**: Verify factual accuracy of claims
- **Relevance**: Assess query-response alignment
- **Metric Aggregation**: Combine multiple metrics

### Observability Stack
- **Langfuse**: Trace visualization and analysis
- **LangSmith**: LLM monitoring and debugging
- **Prometheus**: Metrics collection and export
- **Custom Trace Logger**: Detailed execution traces

### Monitoring
- **Agent Monitor**: Per-agent performance tracking
- **Performance Tracker**: Latency and throughput metrics
- **Error Tracker**: Error aggregation and analysis
- **Alert Manager**: Real-time alerting on thresholds

### Visualization
- **Dashboard Builder**: Create custom dashboards
- **Grafana Integration**: Pre-built Grafana dashboards
- **Chart Generation**: Plotly-based visualizations
- **Report Rendering**: HTML, JSON, Markdown reports

### Multi-Agent Evaluation
- **Coordination Metrics**: Measure agent collaboration
- **Consensus Evaluation**: Assess agreement levels
- **Interaction Tracking**: Monitor agent communications

## Quick Start

### Installation
```bash
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys
```

### Basic Usage
```python
from evaluation.evaluator import Evaluator
from metrics.metric_aggregator import MetricAggregator

# Initialize evaluator
evaluator = Evaluator()

# Evaluate a response
result = evaluator.evaluate_response(
    query="What is machine learning?",
    response="Machine learning is a subset of AI...",
    context="Machine learning enables systems to learn from data..."
)

print(f"Overall Score: {result['overall_score']:.3f}")
print(f"Relevance: {result['relevance']:.3f}")
print(f"Coherence: {result['coherence']:.3f}")
```

### Observability Setup
```python
from observability.langfuse_integration import LangfuseIntegration
from observability.prometheus_exporter import PrometheusExporter

# Initialize observability
langfuse = LangfuseIntegration()
prometheus = PrometheusExporter()

# Trace agent execution
trace_id = langfuse.trace_agent_run(
    agent_name="QuestionAnswering",
    query="What is AI?",
    response="AI is artificial intelligence..."
)

# Record metrics
prometheus.record_agent_request("qa_agent", duration=1.5, status="success")
```

## Examples

Run examples from the `examples/` directory:
```bash
# Basic metrics
python examples/01_basic_metrics.py

# Langfuse tracing
python examples/02_langfuse_tracing.py

# Prometheus monitoring
python examples/03_prometheus_monitoring.py

# Complete evaluation pipeline
python examples/04_evaluation_pipeline.py

# Multi-agent evaluation
python examples/05_multi_agent_evaluation.py

# Dashboard creation
python examples/06_dashboard_creation.py

# Benchmark suite
python examples/07_benchmark_suite.py

# Complete observability
python examples/08_complete_observability.py
```

## Architecture
```
┌─────────────────────────────────────────────────────────┐
│                    AI Agent System                      │
└────────────────────┬────────────────────────────────────┘
                     │
         ┌───────────┴───────────┐
         │                       │
    ┌────▼────┐           ┌─────▼─────┐
    │  Query  │           │  Response │
    └────┬────┘           └─────┬─────┘
         │                      │
         │    ┌─────────────────┘
         │    │
    ┌────▼────▼─────┐
    │  Evaluation   │
    │  • Relevance  │
    │  • Coherence  │
    │  • Grounded   │
    │  • Factual    │
    └────┬──────────┘
         │
    ┌────▼──────────┐
    │ Observability │
    │  • Langfuse   │
    │  • LangSmith  │
    │  • Prometheus │
    │  • Traces     │
    └────┬──────────┘
         │
    ┌────▼──────────┐
    │  Monitoring   │
    │  • Metrics    │
    │  • Errors     │
    │  • Alerts     │
    └────┬──────────┘
         │
    ┌────▼──────────┐
    │Visualization  │
    │  • Dashboards │
    │  • Reports    │
    │  • Charts     │
    └───────────────┘
```

## Configuration

See `configs/` directory for configuration examples:

- `metrics_config.yaml` - Metrics settings
- `langfuse_config.yaml` - Langfuse configuration
- `prometheus_config.yaml` - Prometheus setup
- `grafana_dashboards.json` - Grafana dashboards

## Documentation

Detailed documentation in `docs/`:

- `metrics_guide.md` - Comprehensive metrics guide
- `observability_setup.md` - Setup instructions
- `evaluation_best_practices.md` - Best practices

## Testing
```bash
# Run all tests
pytest tests/

# Run specific test suite
pytest tests/test_metrics.py
pytest tests/test_evaluation.py
pytest tests/test_observability.py

# With coverage
pytest --cov=. tests/
```

## Integration with External Services

### Langfuse

1. Sign up at https://langfuse.com
2. Create a project
3. Copy API keys to `.env`
4. Enable in config: `ENABLE_LANGFUSE=true`

### LangSmith

1. Sign up at https://smith.langchain.com
2. Get API key
3. Add to `.env`: `LANGCHAIN_API_KEY=your_key`
4. Set project: `LANGCHAIN_PROJECT=your_project`

### Prometheus + Grafana

1. Install Prometheus and Grafana
2. Configure Prometheus to scrape metrics
3. Import Grafana dashboards from `configs/`
4. View at http://localhost:3000

## License

MIT License - see LICENSE file for details