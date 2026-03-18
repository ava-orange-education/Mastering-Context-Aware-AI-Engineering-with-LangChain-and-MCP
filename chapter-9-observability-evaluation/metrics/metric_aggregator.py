"""
Aggregate multiple metrics into comprehensive evaluation.
"""

from typing import Dict, Any, List, Optional
import logging
from .groundedness_metric import GroundednessMetric
from .coherence_metric import CoherenceMetric
from .factuality_metric import FactualityMetric
from .relevance_metric import RelevanceMetric

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MetricAggregator:
    """Aggregate multiple evaluation metrics"""
    
    def __init__(self, llm_client=None):
        """
        Initialize metric aggregator
        
        Args:
            llm_client: Optional LLM client for advanced metrics
        """
        self.groundedness = GroundednessMetric(llm_client)
        self.coherence = CoherenceMetric(llm_client)
        self.factuality = FactualityMetric(llm_client)
        self.relevance = RelevanceMetric(llm_client)
    
    def evaluate(self, query: str, response: str, 
                context: Optional[str] = None,
                ground_truth: Optional[str] = None,
                sources: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Comprehensive evaluation with all metrics
        
        Args:
            query: User query
            response: Generated response
            context: Retrieved context
            ground_truth: Known correct answer
            sources: Source documents
            
        Returns:
            Aggregated metrics
        """
        results = {}
        
        # Relevance (query vs response)
        relevance_result = self.relevance.calculate(query, response)
        results['relevance'] = relevance_result['relevance_score']
        results['relevance_details'] = relevance_result
        
        # Coherence (response quality)
        coherence_result = self.coherence.calculate(response)
        results['coherence'] = coherence_result['coherence_score']
        results['coherence_details'] = coherence_result
        
        # Groundedness (if context available)
        if context:
            groundedness_result = self.groundedness.calculate(response, context)
            results['groundedness'] = groundedness_result['groundedness_score']
            results['groundedness_details'] = groundedness_result
        
        # Factuality (if sources or ground truth available)
        if ground_truth or sources:
            factuality_result = self.factuality.calculate(
                response, 
                ground_truth=ground_truth,
                sources=sources
            )
            results['factuality'] = factuality_result['factuality_score']
            results['factuality_details'] = factuality_result
        
        # Calculate overall score
        scores = [results.get('relevance', 0)]
        if 'coherence' in results:
            scores.append(results['coherence'])
        if 'groundedness' in results:
            scores.append(results['groundedness'])
        if 'factuality' in results:
            scores.append(results['factuality'])
        
        results['overall_score'] = sum(scores) / len(scores) if scores else 0.0
        results['metrics_used'] = len(scores)
        
        return results
    
    def evaluate_batch(self, evaluations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluate multiple query-response pairs
        
        Args:
            evaluations: List of dicts with query, response, context, etc.
            
        Returns:
            Aggregated batch results
        """
        individual_results = []
        
        for eval_data in evaluations:
            result = self.evaluate(
                query=eval_data['query'],
                response=eval_data['response'],
                context=eval_data.get('context'),
                ground_truth=eval_data.get('ground_truth'),
                sources=eval_data.get('sources')
            )
            individual_results.append(result)
        
        # Aggregate statistics
        avg_scores = {
            'relevance': 0.0,
            'coherence': 0.0,
            'groundedness': 0.0,
            'factuality': 0.0,
            'overall': 0.0
        }
        
        counts = {key: 0 for key in avg_scores.keys()}
        
        for result in individual_results:
            for key in avg_scores.keys():
                if key == 'overall':
                    avg_scores['overall'] += result.get('overall_score', 0)
                    counts['overall'] += 1
                elif key in result:
                    avg_scores[key] += result[key]
                    counts[key] += 1
        
        # Calculate averages
        for key in avg_scores.keys():
            if counts[key] > 0:
                avg_scores[key] = avg_scores[key] / counts[key]
        
        return {
            'average_scores': avg_scores,
            'individual_results': individual_results,
            'total_evaluations': len(evaluations)
        }