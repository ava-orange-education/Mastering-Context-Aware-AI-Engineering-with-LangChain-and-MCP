"""
Example 4: Evaluation Pipeline
Demonstrates complete evaluation workflow.
"""

import sys
sys.path.append('..')

from evaluation.evaluator import Evaluator
from evaluation.test_sets import TestSetManager
from evaluation.benchmark_runner import BenchmarkRunner
from evaluation.report_generator import ReportGenerator


# Simple mock agent for demonstration
class MockAgent:
    def run(self, query):
        # Simulate agent responses
        responses = {
            "What is the capital of France?": "The capital of France is Paris.",
            "Explain photosynthesis": "Photosynthesis is the process by which plants convert sunlight into energy using chlorophyll.",
            "Compare Python and Java": "Python is interpreted and dynamically typed, while Java is compiled and statically typed.",
            "List three fruits": "Three common fruits are apples, bananas, and oranges."
        }
        
        return {
            'result': responses.get(query, "I don't have information about that."),
            'context': "Sample context from knowledge base..."
        }


def main():
    print("=" * 70)
    print("  EVALUATION PIPELINE EXAMPLE")
    print("=" * 70 + "\n")
    
    # Initialize components
    evaluator = Evaluator()
    test_manager = TestSetManager()
    report_gen = ReportGenerator()
    
    # Create test set
    print("1. CREATING TEST SET")
    print("-" * 70)
    
    test_set = test_manager.create_test_set(
        name="basic_qa",
        description="Basic Q&A evaluation"
    )
    
    test_set.add_test_case(
        query="What is the capital of France?",
        expected_answer="Paris",
        category="factual"
    )
    
    test_set.add_test_case(
        query="Explain photosynthesis",
        expected_answer="Process by which plants convert sunlight into energy",
        category="explanation"
    )
    
    test_set.add_test_case(
        query="Compare Python and Java",
        category="comparison"
    )
    
    test_set.add_test_case(
        query="List three fruits",
        category="enumeration"
    )
    
    print(f"✓ Created test set with {len(test_set.test_cases)} cases")
    print(f"  Categories: {set(tc['category'] for tc in test_set.test_cases)}")
    
    # Evaluate individual response
    print("\n2. EVALUATING SINGLE RESPONSE")
    print("-" * 70)
    
    query = "What is the capital of France?"
    response = "The capital of France is Paris."
    
    result = evaluator.evaluate_response(
        query=query,
        response=response,
        ground_truth="Paris",
        metadata={'category': 'factual'}
    )
    
    print(f"Query: {query}")
    print(f"Response: {response}")
    print(f"\nScores:")
    print(f"  Overall: {result['overall_score']:.3f}")
    print(f"  Relevance: {result['relevance']:.3f}")
    print(f"  Coherence: {result['coherence']:.3f}")
    
    # Evaluate agent on test set
    print("\n3. EVALUATING AGENT ON TEST SET")
    print("-" * 70)
    
    agent = MockAgent()
    
    agent_results = evaluator.evaluate_agent(
        agent=agent,
        test_queries=test_set.get_queries(),
        ground_truths=test_set.get_expected_answers()
    )
    
    summary = agent_results['summary']
    
    print(f"✓ Evaluated {agent_results['total_queries']} queries")
    print(f"\nSummary:")
    print(f"  Average Relevance: {summary['average_relevance']:.3f}")
    print(f"  Average Coherence: {summary['average_coherence']:.3f}")
    print(f"  Average Overall: {summary['average_overall']:.3f}")
    
    # Run benchmark
    print("\n4. RUNNING BENCHMARK")
    print("-" * 70)
    
    benchmark_runner = BenchmarkRunner(evaluator)
    
    benchmark_result = benchmark_runner.run_benchmark(
        agent=agent,
        test_set=test_set,
        name="basic_qa_benchmark"
    )
    
    print(f"✓ Benchmark completed")
    print(f"\nResults:")
    print(f"  Success Rate: {benchmark_result['success_rate']:.1%}")
    print(f"  Total Cases: {benchmark_result['total_cases']}")
    print(f"  Successful: {benchmark_result['successful_cases']}")
    print(f"  Average Latency: {benchmark_result['average_latency_seconds']:.3f}s")
    
    if 'average_metrics' in benchmark_result:
        avg_metrics = benchmark_result['average_metrics']
        print(f"\nAverage Metrics:")
        print(f"  Relevance: {avg_metrics['relevance']:.3f}")
        print(f"  Coherence: {avg_metrics['coherence']:.3f}")
        print(f"  Overall: {avg_metrics['overall']:.3f}")
    
    # Generate report
    print("\n5. GENERATING REPORT")
    print("-" * 70)
    
    # Text report
    text_report = report_gen.generate_text_report(agent_results)
    print("\nText Report Preview:")
    print(text_report[:300] + "...\n")
    
    # Save reports
    report_gen.generate_json_report(agent_results, "evaluation_report.json")
    print("✓ JSON report saved to: evaluation_report.json")
    
    # HTML report
    html_report = report_gen.generate_html_report(agent_results)
    with open("evaluation_report.html", "w") as f:
        f.write(html_report)
    print("✓ HTML report saved to: evaluation_report.html")
    
    # Markdown report
    md_report = report_gen.generate_markdown_report(agent_results)
    with open("evaluation_report.md", "w") as f:
        f.write(md_report)
    print("✓ Markdown report saved to: evaluation_report.md")
    
    print("\n" + "=" * 70)
    print("  EVALUATION PIPELINE DEMONSTRATION COMPLETE")
    print("=" * 70)
    
    print("\n💡 Workflow Summary:")
    print("   1. Create test sets with queries and expected answers")
    print("   2. Evaluate individual responses or entire agents")
    print("   3. Run benchmarks to compare performance")
    print("   4. Generate reports in multiple formats")
    print("   5. Track evaluation history over time")


if __name__ == "__main__":
    main()