"""
Tests for evaluation metrics.
"""

import pytest
from metrics.groundedness_metric import GroundednessMetric
from metrics.coherence_metric import CoherenceMetric
from metrics.factuality_metric import FactualityMetric
from metrics.relevance_metric import RelevanceMetric
from metrics.metric_aggregator import MetricAggregator


class TestGroundednessMetric:
    def test_grounded_response(self):
        """Test response that is well grounded in context"""
        metric = GroundednessMetric()
        
        context = "The capital of France is Paris. Paris is known for the Eiffel Tower."
        response = "Paris is the capital of France and features the Eiffel Tower."
        
        result = metric.calculate(response, context)
        
        assert result['groundedness_score'] > 0.7
        assert result['total_claims'] > 0
    
    def test_ungrounded_response(self):
        """Test response with ungrounded claims"""
        metric = GroundednessMetric()
        
        context = "The capital of France is Paris."
        response = "London is the capital of France and has Big Ben."
        
        result = metric.calculate(response, context)
        
        assert result['groundedness_score'] < 0.5
        assert len(result['ungrounded_claims']) > 0


class TestCoherenceMetric:
    def test_coherent_response(self):
        """Test coherent response"""
        metric = CoherenceMetric()
        
        response = """Machine learning is a subset of AI. It enables computers to learn 
        from data without explicit programming. Therefore, it has many practical applications."""
        
        result = metric.calculate(response)
        
        assert result['coherence_score'] > 0.6
        assert result['structure_score'] > 0
    
    def test_repetitive_response(self):
        """Test response with excessive repetition"""
        metric = CoherenceMetric()
        
        response = """Machine learning is great. Machine learning is great. 
        Machine learning is great. Machine learning is great."""
        
        result = metric.calculate(response)
        
        assert result['repetition_score'] < 0.8


class TestFactualityMetric:
    def test_factual_response(self):
        """Test factually accurate response"""
        metric = FactualityMetric()
        
        sources = ["Water boils at 100 degrees Celsius at sea level."]
        response = "Water boils at 100 degrees Celsius."
        
        result = metric.calculate(response, sources=sources)
        
        assert result['factuality_score'] > 0.5
    
    def test_ground_truth_comparison(self):
        """Test comparison against ground truth"""
        metric = FactualityMetric()
        
        response = "The capital of France is Paris"
        ground_truth = "Paris"
        
        result = metric.calculate(response, ground_truth=ground_truth)
        
        assert 'factuality_score' in result
        assert result['method'] == 'ground_truth_comparison'


class TestRelevanceMetric:
    def test_relevant_response(self):
        """Test relevant response"""
        metric = RelevanceMetric()
        
        query = "What is machine learning?"
        response = "Machine learning is a subset of AI that enables systems to learn from data."
        
        result = metric.calculate(query, response)
        
        assert result['relevance_score'] > 0.5
        assert result['keyword_overlap_score'] > 0
    
    def test_irrelevant_response(self):
        """Test irrelevant response"""
        metric = RelevanceMetric()
        
        query = "What is machine learning?"
        response = "Pizza is delicious and comes in many varieties."
        
        result = metric.calculate(query, response)
        
        assert result['relevance_score'] < 0.5


class TestMetricAggregator:
    def test_aggregation(self):
        """Test metric aggregation"""
        aggregator = MetricAggregator()
        
        query = "What is AI?"
        response = "AI is artificial intelligence, which enables machines to perform tasks."
        context = "Artificial intelligence enables machines to perform intelligent tasks."
        
        result = aggregator.evaluate(query, response, context=context)
        
        assert 'overall_score' in result
        assert 'relevance' in result
        assert 'coherence' in result
        assert 'groundedness' in result
        assert 0 <= result['overall_score'] <= 1