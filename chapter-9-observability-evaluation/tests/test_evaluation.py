"""
Tests for evaluation framework.
"""

import pytest
from evaluation.evaluator import Evaluator
from evaluation.test_sets import TestSet, TestSetManager
from evaluation.benchmark_runner import BenchmarkRunner


class MockAgent:
    def run(self, query):
        return {'result': f"Response to: {query}", 'context': "Test context"}


class TestEvaluator:
    def test_evaluate_response(self):
        """Test single response evaluation"""
        evaluator = Evaluator()
        
        result = evaluator.evaluate_response(
            query="What is AI?",
            response="AI is artificial intelligence.",
            context="Artificial intelligence (AI) refers to machine intelligence."
        )
        
        assert 'overall_score' in result
        assert 'relevance' in result
        assert 'coherence' in result
        assert result['query'] == "What is AI?"
    
    def test_evaluate_agent(self):
        """Test agent evaluation"""
        evaluator = Evaluator()
        agent = MockAgent()
        
        queries = ["What is AI?", "Explain machine learning"]
        
        result = evaluator.evaluate_agent(agent, queries)
        
        assert result['total_queries'] == len(queries)
        assert 'summary' in result
        assert 'individual_results' in result


class TestTestSets:
    def test_create_test_set(self):
        """Test test set creation"""
        test_set = TestSet("test", "Test set")
        
        test_set.add_test_case(
            query="What is AI?",
            expected_answer="Artificial intelligence",
            category="factual"
        )
        
        assert len(test_set.test_cases) == 1
        assert test_set.get_queries()[0] == "What is AI?"
    
    def test_test_set_manager(self):
        """Test test set manager"""
        manager = TestSetManager()
        
        test_set = manager.create_test_set("rag_test", "RAG tests")
        test_set.add_test_case("Query 1", "Answer 1")
        
        retrieved = manager.get_test_set("rag_test")
        
        assert retrieved is not None
        assert len(retrieved.test_cases) == 1


class TestBenchmarkRunner:
    def test_run_benchmark(self):
        """Test benchmark execution"""
        evaluator = Evaluator()
        runner = BenchmarkRunner(evaluator)
        agent = MockAgent()
        
        test_set = TestSet("bench", "Benchmark")
        test_set.add_test_case("Query 1", "Answer 1")
        test_set.add_test_case("Query 2", "Answer 2")
        
        result = runner.run_benchmark(agent, test_set, "test_bench")
        
        assert 'total_cases' in result
        assert 'success_rate' in result
        assert result['total_cases'] == 2