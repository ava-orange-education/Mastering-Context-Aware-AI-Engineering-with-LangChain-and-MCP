# Evaluation Metrics Guide

This guide explains the evaluation metrics available in the framework and how to use them effectively.

## Available Metrics

### 1. Relevance

Measures how well the response addresses the user's query.

**Calculation:**
- Keyword overlap between query and response
- Intent matching (definition, comparison, enumeration, etc.)

**Score Range:** 0.0 to 1.0 (higher is better)

**Usage:**
```python
from metrics.relevance_metric import RelevanceMetric

metric = RelevanceMetric()
result = metric.calculate(query="What is AI?", response="AI is artificial intelligence...")

print(f"Relevance: {result['relevance_score']}")
```

**Best Practices:**
- Use for query-response pair evaluation
- Combine with other metrics for comprehensive assessment
- Consider domain-specific query patterns

### 2. Coherence

Measures logical flow and consistency within the response.

**Calculation:**
- Structure quality (sentence length variation)
- Repetition detection
- Transition word usage

**Score Range:** 0.0 to 1.0 (higher is better)

**Usage:**
```python
from metrics.coherence_metric import CoherenceMetric

metric = CoherenceMetric()
result = metric.calculate(response="Your response text...")

print(f"Coherence: {result['coherence_score']}")
print(f"Structure: {result['structure_score']}")
```

**Best Practices:**
- Use for longer responses (>50 words)
- Check individual sub-scores for diagnosis
- Set thresholds based on your use case

### 3. Groundedness

Measures how well the response is grounded in provided context.

**Calculation:**
- Claim extraction from response
- Keyword overlap with context
- Optional LLM-based verification

**Score Range:** 0.0 to 1.0 (higher is better)

**Usage:**
```python
from metrics.groundedness_metric import GroundednessMetric

metric = GroundednessMetric()
result = metric.calculate(
    response="Your response...",
    context="Source context..."
)

print(f"Groundedness: {result['groundedness_score']}")
print(f"Grounded claims: {result['grounded_claims']}/{result['total_claims']}")
```

**Best Practices:**
- Always provide comprehensive context
- Review ungrounded claims for false positives
- Use LLM verification for critical applications

### 4. Factuality

Measures factual accuracy of claims in the response.

**Calculation:**
- Comparison against ground truth (if available)
- Verification against source documents
- Hallucination pattern detection

**Score Range:** 0.0 to 1.0 (higher is better)

**Usage:**
```python
from metrics.factuality_metric import FactualityMetric

metric = FactualityMetric()

# With ground truth
result = metric.calculate(
    response="Paris is the capital of France",
    ground_truth="Paris"
)

# With sources
result = metric.calculate(
    response="Water boils at 100°C",
    sources=["Water boils at 100 degrees Celsius at sea level"]
)

print(f"Factuality: {result['factuality_score']}")
```

**Best Practices:**
- Provide either ground truth or sources
- Use multiple sources for verification
- Consider domain expertise for interpretation

## Metric Aggregation

Combine multiple metrics for comprehensive evaluation:
```python
from metrics.metric_aggregator import MetricAggregator

aggregator = MetricAggregator()

result = aggregator.evaluate(
    query="What is machine learning?",
    response="Machine learning is...",
    context="Source context...",
    sources=["Source 1", "Source 2"]
)

print(f"Overall Score: {result['overall_score']}")
print(f"Relevance: {result['relevance']}")
print(f"Coherence: {result['coherence']}")
print(f"Groundedness: {result['groundedness']}")
print(f"Factuality: {result['factuality']}")
```

## Customizing Metrics

### Setting Thresholds
```python
# In your configuration
THRESHOLDS = {
    'relevance': 0.7,
    'coherence': 0.8,
    'groundedness': 0.75,
    'factuality': 0.85
}

# Check against thresholds
passes = all(
    result[metric] >= THRESHOLDS[metric]
    for metric in THRESHOLDS
    if metric in result
)
```

### Weighted Aggregation
```python
weights = {
    'relevance': 0.3,
    'coherence': 0.2,
    'groundedness': 0.3,
    'factuality': 0.2
}

weighted_score = sum(
    result[metric] * weights[metric]
    for metric in weights
    if metric in result
)
```

## Interpreting Results

### Score Ranges

- **0.9 - 1.0**: Excellent
- **0.7 - 0.9**: Good
- **0.5 - 0.7**: Acceptable
- **0.0 - 0.5**: Needs improvement

### Common Issues

**Low Relevance:**
- Query not properly understood
- Response off-topic
- Missing key information

**Low Coherence:**
- Repetitive content
- Poor logical flow
- Lack of transitions

**Low Groundedness:**
- Claims not supported by context
- Missing source information
- Hallucinated facts

**Low Factuality:**
- Incorrect information
- Contradicts sources
- Fabricated details

## Best Practices

1. **Use Multiple Metrics**: Don't rely on a single metric
2. **Provide Context**: Always include relevant context for groundedness
3. **Set Realistic Thresholds**: Based on your domain and requirements
4. **Monitor Trends**: Track metrics over time
5. **Validate Results**: Manually review low-scoring outputs
6. **Iterate**: Use metrics to improve prompts and systems

## Examples

See `examples/01_basic_metrics.py` for complete working examples.