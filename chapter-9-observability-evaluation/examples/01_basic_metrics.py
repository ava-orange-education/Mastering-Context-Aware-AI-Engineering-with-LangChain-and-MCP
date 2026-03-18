"""
Example 1: Basic Metrics
Demonstrates basic evaluation metrics.
"""

import sys
sys.path.append('..')

from metrics.groundedness_metric import GroundednessMetric
from metrics.coherence_metric import CoherenceMetric
from metrics.factuality_metric import FactualityMetric
from metrics.relevance_metric import RelevanceMetric
from metrics.metric_aggregator import MetricAggregator


def main():
    print("=" * 70)
    print("  BASIC EVALUATION METRICS EXAMPLE")
    print("=" * 70 + "\n")
    
    # Test query and response
    query = "What are the benefits of regular exercise?"
    
    response = """Regular exercise provides numerous health benefits. It improves 
cardiovascular health, strengthens muscles, and enhances mental well-being. 
Studies show that people who exercise regularly have lower rates of chronic 
diseases. Exercise also helps with weight management and improves sleep quality."""
    
    context = """Research on Exercise Benefits:
Exercise improves cardiovascular health by strengthening the heart muscle.
Regular physical activity helps build and maintain strong muscles and bones.
Exercise has been shown to reduce symptoms of depression and anxiety.
People who maintain regular exercise routines have lower incidence of type 2 diabetes,
heart disease, and certain cancers. Exercise aids in weight control by burning calories
and increasing metabolism. Physical activity can improve sleep quality and duration."""
    
    print("Query:", query)
    print("\nResponse:", response[:100] + "...")
    print("\n" + "-" * 70)
    
    # 1. Relevance Metric
    print("\n1. RELEVANCE METRIC")
    print("-" * 70)
    
    relevance_metric = RelevanceMetric()
    relevance_result = relevance_metric.calculate(query, response)
    
    print(f"Relevance Score: {relevance_result['relevance_score']:.3f}")
    print(f"Keyword Overlap: {relevance_result['keyword_overlap_score']:.3f}")
    print(f"Intent Match: {relevance_result['intent_match_score']:.3f}")
    
    # 2. Coherence Metric
    print("\n2. COHERENCE METRIC")
    print("-" * 70)
    
    coherence_metric = CoherenceMetric()
    coherence_result = coherence_metric.calculate(response)
    
    print(f"Coherence Score: {coherence_result['coherence_score']:.3f}")
    print(f"Structure Score: {coherence_result['structure_score']:.3f}")
    print(f"Repetition Score: {coherence_result['repetition_score']:.3f}")
    print(f"Transition Score: {coherence_result['transition_score']:.3f}")
    
    # 3. Groundedness Metric
    print("\n3. GROUNDEDNESS METRIC")
    print("-" * 70)
    
    groundedness_metric = GroundednessMetric()
    groundedness_result = groundedness_metric.calculate(response, context)
    
    print(f"Groundedness Score: {groundedness_result['groundedness_score']:.3f}")
    print(f"Grounded Claims: {groundedness_result['grounded_claims']}/{groundedness_result['total_claims']}")
    
    if groundedness_result['ungrounded_claims']:
        print("\nUngrounded claims:")
        for claim in groundedness_result['ungrounded_claims'][:2]:
            print(f"  - {claim}")
    
    # 4. Factuality Metric
    print("\n4. FACTUALITY METRIC")
    print("-" * 70)
    
    factuality_metric = FactualityMetric()
    factuality_result = factuality_metric.calculate(response, sources=[context])
    
    print(f"Factuality Score: {factuality_result['factuality_score']:.3f}")
    print(f"Verified Claims: {factuality_result['verified_claims']}/{factuality_result['total_claims']}")
    
    # 5. Aggregated Metrics
    print("\n5. AGGREGATED METRICS")
    print("-" * 70)
    
    aggregator = MetricAggregator()
    aggregated = aggregator.evaluate(
        query=query,
        response=response,
        context=context,
        sources=[context]
    )
    
    print(f"Overall Score: {aggregated['overall_score']:.3f}")
    print(f"\nBreakdown:")
    print(f"  Relevance:    {aggregated['relevance']:.3f}")
    print(f"  Coherence:    {aggregated['coherence']:.3f}")
    print(f"  Groundedness: {aggregated['groundedness']:.3f}")
    print(f"  Factuality:   {aggregated['factuality']:.3f}")
    
    print("\n" + "=" * 70)
    print("  METRICS DEMONSTRATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()