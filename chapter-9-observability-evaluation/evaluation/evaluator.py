"""
Main evaluation orchestrator for AI agents.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
import logging
from ..metrics.metric_aggregator import MetricAggregator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Evaluator:
    """Orchestrate evaluation of AI agent responses"""
    
    def __init__(self, llm_client=None, config: Optional[Dict] = None):
        """
        Initialize evaluator
        
        Args:
            llm_client: LLM client for advanced metrics
            config: Evaluation configuration
        """
        self.llm = llm_client
        self.config = config or {}
        self.metric_aggregator = MetricAggregator(llm_client)
        self.evaluation_history: List[Dict] = []
    
    def evaluate_response(self, query: str, response: str,
                         context: Optional[str] = None,
                         ground_truth: Optional[str] = None,
                         metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Evaluate a single response
        
        Args:
            query: User query
            response: Agent response
            context: Retrieved context
            ground_truth: Known correct answer
            metadata: Additional metadata
            
        Returns:
            Evaluation results
        """
        logger.info(f"Evaluating response for query: {query[:50]}...")
        
        # Run metrics
        results = self.metric_aggregator.evaluate(
            query=query,
            response=response,
            context=context,
            ground_truth=ground_truth
        )
        
        # Add metadata
        results['query'] = query
        results['response'] = response
        results['timestamp'] = datetime.now().isoformat()
        results['metadata'] = metadata or {}
        
        # Store in history
        self.evaluation_history.append(results)
        
        logger.info(f"Evaluation complete. Overall score: {results['overall_score']:.2f}")
        
        return results
    
    def evaluate_agent(self, agent, test_queries: List[str],
                      ground_truths: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Evaluate agent on multiple test queries
        
        Args:
            agent: Agent to evaluate
            test_queries: List of test queries
            ground_truths: Optional list of correct answers
            
        Returns:
            Comprehensive evaluation results
        """
        logger.info(f"Evaluating agent on {len(test_queries)} queries")
        
        results = []
        
        for idx, query in enumerate(test_queries):
            # Generate response
            try:
                agent_result = agent.run(query)
                response = agent_result.get('result', agent_result.get('response', ''))
                context = agent_result.get('context', '')
            except Exception as e:
                logger.error(f"Agent failed on query {idx}: {e}")
                response = ""
                context = ""
            
            # Get ground truth if available
            ground_truth = ground_truths[idx] if ground_truths and idx < len(ground_truths) else None
            
            # Evaluate
            eval_result = self.evaluate_response(
                query=query,
                response=response,
                context=context,
                ground_truth=ground_truth,
                metadata={'test_index': idx}
            )
            
            results.append(eval_result)
        
        # Aggregate results
        summary = self._create_summary(results)
        
        return {
            'summary': summary,
            'individual_results': results,
            'total_queries': len(test_queries),
            'evaluation_time': datetime.now().isoformat()
        }
    
    def _create_summary(self, results: List[Dict]) -> Dict[str, Any]:
        """Create summary statistics from results"""
        if not results:
            return {}
        
        # Calculate averages
        avg_relevance = sum(r.get('relevance', 0) for r in results) / len(results)
        avg_coherence = sum(r.get('coherence', 0) for r in results) / len(results)
        avg_overall = sum(r.get('overall_score', 0) for r in results) / len(results)
        
        # Count metrics availability
        groundedness_count = sum(1 for r in results if 'groundedness' in r)
        factuality_count = sum(1 for r in results if 'factuality' in r)
        
        avg_groundedness = (sum(r.get('groundedness', 0) for r in results) / groundedness_count 
                           if groundedness_count > 0 else None)
        avg_factuality = (sum(r.get('factuality', 0) for r in results) / factuality_count 
                         if factuality_count > 0 else None)
        
        summary = {
            'average_relevance': avg_relevance,
            'average_coherence': avg_coherence,
            'average_overall': avg_overall,
            'total_evaluations': len(results)
        }
        
        if avg_groundedness is not None:
            summary['average_groundedness'] = avg_groundedness
        if avg_factuality is not None:
            summary['average_factuality'] = avg_factuality
        
        return summary
    
    def get_evaluation_report(self) -> Dict[str, Any]:
        """Generate comprehensive evaluation report"""
        if not self.evaluation_history:
            return {'error': 'No evaluations performed yet'}
        
        summary = self._create_summary(self.evaluation_history)
        
        # Group by metadata if available
        by_category = {}
        for eval_result in self.evaluation_history:
            category = eval_result.get('metadata', {}).get('category', 'default')
            if category not in by_category:
                by_category[category] = []
            by_category[category].append(eval_result)
        
        category_summaries = {}
        for category, evals in by_category.items():
            category_summaries[category] = self._create_summary(evals)
        
        return {
            'overall_summary': summary,
            'by_category': category_summaries,
            'total_evaluations': len(self.evaluation_history),
            'history': self.evaluation_history
        }