# Evaluation Best Practices

Best practices for evaluating AI agents effectively.

## Test Set Design

### 1. Coverage

Ensure test sets cover:
- **Different query types**: Questions, commands, comparisons
- **Difficulty levels**: Simple, medium, complex
- **Domain areas**: All topics your agent handles
- **Edge cases**: Ambiguous queries, rare scenarios

Example:
```python
test_set.add_test_case(
    query="What is AI?",  # Simple factual
    category="factual"
)

test_set.add_test_case(
    query="Compare supervised and unsupervised learning",  # Complex comparison
    category="comparison"
)
```

### 2. Ground Truth

- Provide expected answers when possible
- Include source documents for verification
- Document acceptable variations
- Update ground truth regularly

### 3. Realistic Data

- Use real user queries
- Include typos and variations
- Test multilingual if applicable
- Consider context variations

## Evaluation Strategy

### 1. Metric Selection

Choose metrics based on your use case:

**Customer Support:**
- Relevance (primary)
- Coherence
- Factuality

**Content Generation:**
- Coherence (primary)
- Relevance
- Creativity (custom metric)

**Research/Analysis:**
- Factuality (primary)
- Groundedness
- Completeness

### 2. Thresholds

Set realistic thresholds:
```python
# Production thresholds
THRESHOLDS = {
    'overall': 0.8,      # Must be high quality
    'relevance': 0.85,   # Must address query
    'factuality': 0.90,  # Must be accurate
    'coherence': 0.75    # Can be less strict
}
```

### 3. Frequency

- **Development**: Every code change
- **Staging**: Daily
- **Production**: Continuous sampling (1-10%)

## Benchmark Design

### 1. Comparison Benchmarks

Compare against:
- Baseline model
- Previous version
- Competitors
- Human performance
```python
results = {
    'baseline': run_benchmark(baseline_agent, test_set),
    'current': run_benchmark(current_agent, test_set),
    'competitor': run_benchmark(competitor_agent, test_set)
}
```

### 2. A/B Testing
```python
# Split traffic
variant_a_results = evaluate_variant_a(queries)
variant_b_results = evaluate_variant_b(queries)

# Statistical significance
from scipy.stats import ttest_ind
statistic, pvalue = ttest_ind(
    variant_a_results,
    variant_b_results
)
```

### 3. Regression Testing

Track metrics over time:
```python
import matplotlib.pyplot as plt

dates = [...]
scores = [...]

plt.plot(dates, scores)
plt.axhline(y=0.8, color='r', linestyle='--', label='Threshold')
plt.title('Evaluation Scores Over Time')
plt.show()
```

## Continuous Evaluation

### 1. Production Sampling
```python
import random

def should_evaluate(sampling_rate=0.01):
    return random.random() < sampling_rate

# In production
if should_evaluate():
    evaluation_result = evaluator.evaluate_response(
        query=query,
        response=response
    )
    log_to_monitoring(evaluation_result)
```

### 2. User Feedback

Collect implicit and explicit feedback:
```python
# Explicit feedback
thumbs_up = user_clicked_thumbs_up()

# Implicit feedback
followed_suggestion = user_clicked_result()
time_to_reformulate = query_reformulated_in_seconds()

# Combine with automated metrics
combined_score = (
    0.5 * automated_score +
    0.3 * thumbs_up_rate +
    0.2 * click_through_rate
)
```

### 3. Drift Detection

Monitor for distribution shifts:
```python
from scipy.stats import ks_2samp

# Compare recent vs baseline
recent_scores = get_recent_scores(days=7)
baseline_scores = get_baseline_scores()

statistic, pvalue = ks_2samp(recent_scores, baseline_scores)

if pvalue < 0.05:
    alert("Significant distribution shift detected!")
```

## Multi-Agent Evaluation

### 1. Coordination Metrics
```python
from multi_agent_eval.coordination_metrics import CoordinationMetrics

coord = CoordinationMetrics()

# Track interactions
coord.record_interaction("agent_a", "agent_b", "request")
coord.record_interaction("agent_b", "agent_a", "response")

# Calculate score
score = coord.calculate_coordination_score(["agent_a", "agent_b"])
```

### 2. Consensus Evaluation
```python
from multi_agent_eval.consensus_evaluator import ConsensusEvaluator

evaluator = ConsensusEvaluator()

agent_responses = {
    "agent_a": "Response A",
    "agent_b": "Response B",
    "agent_c": "Response C"
}

consensus = evaluator.evaluate_consensus(agent_responses)
print(f"Consensus score: {consensus['consensus_score']}")
```

## Reporting

### 1. Regular Reports

Generate weekly/monthly reports:
```python
from evaluation.report_generator import ReportGenerator

generator = ReportGenerator()

# Generate HTML report
html = generator.generate_html_report(results)

# Send to stakeholders
send_email(
    to="team@company.com",
    subject="Weekly Evaluation Report",
    body=html
)
```

### 2. Dashboards

Create live dashboards:
- Current performance metrics
- Trend analysis
- Comparison to baselines
- Error breakdowns

### 3. Alerts

Set up alerts for:
- Score drops below threshold
- Sudden increase in errors
- Performance degradation
- Cost anomalies

## Common Pitfalls

### 1. Overfitting to Test Set

**Problem**: Model performs well on test set but poorly in production

**Solution**:
- Regularly update test sets
- Use holdout sets
- Monitor production metrics
- Collect real user data

### 2. Ignoring Edge Cases

**Problem**: Rare scenarios cause failures

**Solution**:
- Include edge cases in test sets
- Monitor for unusual queries
- Implement fallback behaviors
- Log all failures for analysis

### 3. Metric Gaming

**Problem**: Optimizing for metrics at expense of real quality

**Solution**:
- Use multiple complementary metrics
- Include human evaluation
- Monitor user satisfaction
- Review individual examples

### 4. Stale Evaluations

**Problem**: Evaluation data becomes outdated

**Solution**:
- Refresh test sets quarterly
- Update ground truth regularly
- Monitor domain drift
- Incorporate new use cases

## Tools and Automation

### 1. CI/CD Integration
```yaml
# .github/workflows/evaluation.yml
name: Evaluation

on: [push]

jobs:
  evaluate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Run evaluation
        run: python -m pytest tests/
      - name: Upload results
        uses: actions/upload-artifact@v2
        with:
          name: evaluation-results
          path: results/
```

### 2. Automated Reporting
```python
# Schedule daily reports
from apscheduler.schedulers.background import BackgroundScheduler

scheduler = BackgroundScheduler()

def daily_report():
    results = run_daily_evaluation()
    report = generate_report(results)
    send_to_slack(report)

scheduler.add_job(daily_report, 'cron', hour=9)
scheduler.start()
```

### 3. Monitoring Integration
```python
# Send metrics to monitoring
prometheus.record_evaluation_score("relevance", score)
langfuse.log_score(trace_id, "overall", overall_score)
```

## Resources

- Examples: `examples/04_evaluation_pipeline.py`
- Metrics: `docs/metrics_guide.md`
- Observability: `docs/observability_setup.md`