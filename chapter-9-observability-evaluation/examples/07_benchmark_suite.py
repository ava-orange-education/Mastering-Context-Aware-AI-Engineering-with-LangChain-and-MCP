"""
Example 7: Benchmark Suite
Demonstrates comprehensive benchmarking.
"""

import sys
sys.path.append('..')

from evaluation.evaluator import Evaluator
from evaluation.test_sets import TestSetManager
from evaluation.benchmark_runner import BenchmarkRunner
import time


# Mock agents for comparison
class FastAgent:
    def run(self, query):
        time.sleep(0.1)  # Fast response
        return {'result': f"Fast response to: {query}", 'context': "Context..."}


class AccurateAgent:
    def run(self, query):
        time.sleep(0.5)  # Slower but more thorough
        return {'result': f"Detailed and accurate response to: {query}", 'context': "Comprehensive context..."}


class BalancedAgent:
    def run(self, query):
        time.sleep(0.3)  # Balanced
        return {'result': f"Balanced response to: {query}", 'context': "Good context..."}


def main():
    print("=" * 70)
    print("  BENCHMARK SUITE EXAMPLE")
    print("=" * 70 + "\n")
    
    # Initialize
    evaluator = Evaluator()
    test_manager = TestSetManager()
    benchmark_runner = BenchmarkRunner(evaluator)
    
    # Create test sets
    print("1. CREATING TEST SETS")
    print("-" * 70)
    
    # RAG test set
    rag_tests = test_manager.create_rag_test_set()
    print(f"✓ RAG test set: {len(rag_tests.test_cases)} cases")
    
    # Multi-agent test set
    ma_tests = test_manager.create_multi_agent_test_set()
    print(f"✓ Multi-agent test set: {len(ma_tests.test_cases)} cases")
    
    # Initialize agents
    print("\n2. INITIALIZING AGENTS")
    print("-" * 70)
    
    agents = {
        'FastAgent': FastAgent(),
        'AccurateAgent': AccurateAgent(),
        'BalancedAgent': BalancedAgent()
    }
    
    for name in agents.keys():
        print(f"  • {name}")
    
    # Run benchmarks
    print("\n3. RUNNING BENCHMARKS")
    print("-" * 70)
    
    results = {}
    
    for agent_name, agent in agents.items():
        print(f"\nBenchmarking {agent_name}...")
        
        result = benchmark_runner.run_benchmark(
            agent=agent,
            test_set=rag_tests,
            name=f"{agent_name}_benchmark"
        )
        
        results[agent_name] = result
        
        print(f"  Success Rate: {result['success_rate']:.1%}")
        print(f"  Avg Latency: {result['average_latency_seconds']:.3f}s")
        
        if 'average_metrics' in result:
            print(f"  Overall Score: {result['average_metrics']['overall']:.3f}")
    
    # Compare benchmarks
    print("\n4. BENCHMARK COMPARISON")
    print("-" * 70)
    
    comparison = benchmark_runner.compare_benchmarks()
    
    print("\nAll Benchmarks:")
    for bench in comparison['benchmarks']:
        print(f"\n{bench['name']}:")
        print(f"  Success Rate: {bench['success_rate']:.1%}")
        print(f"  Avg Latency: {bench['average_latency']:.3f}s")
        
        if 'average_overall_score' in bench:
            print(f"  Overall Score: {bench['average_overall_score']:.3f}")
    
    print("\n\nBest Performers:")
    print(f"  Best Score: {comparison['best_overall_score']['benchmark']} "
          f"({comparison['best_overall_score']['score']:.3f})")
    print(f"  Fastest: {comparison['fastest_average_latency']['benchmark']} "
          f"({comparison['fastest_average_latency']['latency_seconds']:.3f}s)")
    
    # Detailed analysis
    print("\n5. DETAILED ANALYSIS")
    print("-" * 70)
    
    print("\nLatency vs Quality Tradeoff:")
    print("-" * 70)
    print(f"{'Agent':<15} {'Latency':<12} {'Quality':<10} {'Efficiency':<10}")
    print("-" * 70)
    
    for agent_name, result in results.items():
        latency = result['average_latency_seconds']
        quality = result.get('average_metrics', {}).get('overall', 0)
        
        # Efficiency = quality / latency (higher is better)
        efficiency = quality / latency if latency > 0 else 0
        
        print(f"{agent_name:<15} {latency:<12.3f} {quality:<10.3f} {efficiency:<10.3f}")
    
    # Performance categories
    print("\n6. PERFORMANCE CATEGORIES")
    print("-" * 70)
    
    for agent_name, result in results.items():
        latency = result['average_latency_seconds']
        success_rate = result['success_rate']
        
        # Categorize
        if latency < 0.2 and success_rate > 0.9:
            category = "⚡ Fast & Reliable"
        elif latency < 0.4 and success_rate > 0.95:
            category = "⚖️  Balanced"
        elif success_rate > 0.98:
            category = "🎯 Highly Accurate"
        else:
            category = "📊 Standard"
        
        print(f"{agent_name}: {category}")
    
    print("\n" + "=" * 70)
    print("  BENCHMARK SUITE DEMONSTRATION COMPLETE")
    print("=" * 70)
    
    print("\n💡 Benchmarking Best Practices:")
    print("   • Use diverse test sets covering different scenarios")
    print("   • Run multiple iterations for statistical significance")
    print("   • Consider latency-quality tradeoffs")
    print("   • Track benchmarks over time to detect regressions")
    print("   • Compare against baselines and competitors")


if __name__ == "__main__":
    main()