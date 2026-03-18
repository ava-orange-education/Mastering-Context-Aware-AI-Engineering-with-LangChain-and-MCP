"""
Run evaluation benchmarks on AI agents.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
import time
import logging
from .evaluator import Evaluator
from .test_sets import TestSet

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BenchmarkRunner:
    """Run comprehensive benchmarks on agents"""
    
    def __init__(self, evaluator: Evaluator):
        """
        Initialize benchmark runner
        
        Args:
            evaluator: Evaluator instance
        """
        self.evaluator = evaluator
        self.benchmark_results: List[Dict] = []
    
    def run_benchmark(self, agent, test_set: TestSet, 
                     name: str = "benchmark") -> Dict[str, Any]:
        """
        Run benchmark on agent with test set
        
        Args:
            agent: Agent to benchmark
            test_set: Test set to use
            name: Benchmark name
            
        Returns:
            Benchmark results
        """
        logger.info(f"Running benchmark '{name}' with {len(test_set.test_cases)} test cases")
        
        start_time = time.time()
        
        results = []
        latencies = []
        
        for idx, test_case in enumerate(test_set.test_cases):
            query = test_case['query']
            expected = test_case.get('expected_answer')
            context = test_case.get('context')
            
            # Measure latency
            case_start = time.time()
            
            try:
                # Run agent
                agent_result = agent.run(query)
                response = agent_result.get('result', agent_result.get('response', ''))
                retrieved_context = agent_result.get('context', context or '')
                
                case_latency = time.time() - case_start
                latencies.append(case_latency)
                
                # Evaluate
                eval_result = self.evaluator.evaluate_response(
                    query=query,
                    response=response,
                    context=retrieved_context,
                    ground_truth=expected,
                    metadata={
                        'test_case_index': idx,
                        'category': test_case.get('category'),
                        'latency_seconds': case_latency
                    }
                )
                
                eval_result['success'] = True
                eval_result['latency'] = case_latency
                
            except Exception as e:
                logger.error(f"Benchmark failed on test case {idx}: {e}")
                eval_result = {
                    'query': query,
                    'success': False,
                    'error': str(e),
                    'latency': time.time() - case_start
                }
                latencies.append(eval_result['latency'])
            
            results.append(eval_result)
        
        total_time = time.time() - start_time
        
        # Calculate statistics
        successful = [r for r in results if r.get('success', False)]
        success_rate = len(successful) / len(results) if results else 0
        
        avg_latency = sum(latencies) / len(latencies) if latencies else 0
        
        benchmark_result = {
            'benchmark_name': name,
            'test_set_name': test_set.name,
            'total_cases': len(results),
            'successful_cases': len(successful),
            'success_rate': success_rate,
            'total_time_seconds': total_time,
            'average_latency_seconds': avg_latency,
            'min_latency_seconds': min(latencies) if latencies else 0,
            'max_latency_seconds': max(latencies) if latencies else 0,
            'timestamp': datetime.now().isoformat(),
            'individual_results': results
        }
        
        # Add metric averages for successful cases
        if successful:
            avg_relevance = sum(r.get('relevance', 0) for r in successful) / len(successful)
            avg_coherence = sum(r.get('coherence', 0) for r in successful) / len(successful)
            avg_overall = sum(r.get('overall_score', 0) for r in successful) / len(successful)
            
            benchmark_result['average_metrics'] = {
                'relevance': avg_relevance,
                'coherence': avg_coherence,
                'overall': avg_overall
            }
        
        self.benchmark_results.append(benchmark_result)
        
        logger.info(f"Benchmark complete. Success rate: {success_rate:.2%}, Avg latency: {avg_latency:.2f}s")
        
        return benchmark_result
    
    def compare_benchmarks(self, benchmark_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Compare multiple benchmark runs
        
        Args:
            benchmark_names: Specific benchmarks to compare (None for all)
            
        Returns:
            Comparison results
        """
        if not self.benchmark_results:
            return {'error': 'No benchmarks to compare'}
        
        results_to_compare = self.benchmark_results
        
        if benchmark_names:
            results_to_compare = [
                r for r in self.benchmark_results 
                if r['benchmark_name'] in benchmark_names
            ]
        
        comparison = {
            'benchmarks': [],
            'best_overall_score': None,
            'fastest_average_latency': None
        }
        
        best_score = 0
        best_score_name = None
        fastest_latency = float('inf')
        fastest_name = None
        
        for result in results_to_compare:
            summary = {
                'name': result['benchmark_name'],
                'success_rate': result['success_rate'],
                'average_latency': result['average_latency_seconds'],
                'total_cases': result['total_cases']
            }
            
            if 'average_metrics' in result:
                summary['average_overall_score'] = result['average_metrics']['overall']
                
                if result['average_metrics']['overall'] > best_score:
                    best_score = result['average_metrics']['overall']
                    best_score_name = result['benchmark_name']
            
            if result['average_latency_seconds'] < fastest_latency:
                fastest_latency = result['average_latency_seconds']
                fastest_name = result['benchmark_name']
            
            comparison['benchmarks'].append(summary)
        
        comparison['best_overall_score'] = {
            'benchmark': best_score_name,
            'score': best_score
        }
        
        comparison['fastest_average_latency'] = {
            'benchmark': fastest_name,
            'latency_seconds': fastest_latency
        }
        
        return comparison