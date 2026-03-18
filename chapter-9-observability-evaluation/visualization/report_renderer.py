"""
Evaluate consensus among multiple agents.
"""

from typing import Dict, Any, List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ConsensusEvaluator:
    """Evaluate consensus and agreement among agents"""
    
    def __init__(self):
        """Initialize consensus evaluator"""
        pass
    
    def evaluate_consensus(self, agent_responses: Dict[str, str]) -> Dict[str, Any]:
        """
        Evaluate consensus among agent responses
        
        Args:
            agent_responses: Dict mapping agent names to responses
            
        Returns:
            Consensus metrics
        """
        if len(agent_responses) < 2:
            return {
                'consensus_score': 1.0,
                'agreement_level': 'insufficient_agents'
            }
        
        responses = list(agent_responses.values())
        
        # Calculate pairwise similarity
        similarities = []
        for i in range(len(responses)):
            for j in range(i + 1, len(responses)):
                similarity = self._calculate_similarity(responses[i], responses[j])
                similarities.append(similarity)
        
        # Average similarity = consensus score
        consensus_score = sum(similarities) / len(similarities) if similarities else 0
        
        # Determine agreement level
        if consensus_score >= 0.8:
            agreement_level = 'strong_consensus'
        elif consensus_score >= 0.6:
            agreement_level = 'moderate_consensus'
        elif consensus_score >= 0.4:
            agreement_level = 'weak_consensus'
        else:
            agreement_level = 'no_consensus'
        
        return {
            'consensus_score': consensus_score,
            'agreement_level': agreement_level,
            'num_agents': len(agent_responses),
            'pairwise_similarities': similarities
        }
    
    def find_outliers(self, agent_responses: Dict[str, str]) -> List[str]:
        """
        Find agents with outlier responses
        
        Args:
            agent_responses: Dict mapping agent names to responses
            
        Returns:
            List of outlier agent names
        """
        if len(agent_responses) < 3:
            return []
        
        # Calculate average similarity for each agent to others
        agent_similarities = {}
        
        for agent1, response1 in agent_responses.items():
            similarities = []
            
            for agent2, response2 in agent_responses.items():
                if agent1 != agent2:
                    sim = self._calculate_similarity(response1, response2)
                    similarities.append(sim)
            
            avg_similarity = sum(similarities) / len(similarities)
            agent_similarities[agent1] = avg_similarity
        
        # Find agents with below-average similarity
        overall_avg = sum(agent_similarities.values()) / len(agent_similarities)
        threshold = overall_avg * 0.7  # 30% below average
        
        outliers = [
            agent for agent, sim in agent_similarities.items()
            if sim < threshold
        ]
        
        return outliers
    
    def evaluate_decision_quality(self, agent_decisions: Dict[str, Any],
                                  ground_truth: Optional[Any] = None) -> Dict[str, Any]:
        """
        Evaluate quality of agent decisions
        
        Args:
            agent_decisions: Dict mapping agent names to decisions
            ground_truth: Optional correct decision
            
        Returns:
            Decision quality metrics
        """
        if not agent_decisions:
            return {'decision_quality': 0.0}
        
        # If ground truth available, calculate accuracy
        if ground_truth is not None:
            correct = sum(
                1 for decision in agent_decisions.values()
                if decision == ground_truth
            )
            
            accuracy = correct / len(agent_decisions)
            
            return {
                'decision_quality': accuracy,
                'correct_agents': correct,
                'total_agents': len(agent_decisions),
                'accuracy': accuracy
            }
        
        # Without ground truth, use majority voting
        from collections import Counter
        decision_counts = Counter(agent_decisions.values())
        
        if not decision_counts:
            return {'decision_quality': 0.0}
        
        most_common_count = decision_counts.most_common(1)[0][1]
        majority_score = most_common_count / len(agent_decisions)
        
        return {
            'decision_quality': majority_score,
            'majority_decision': decision_counts.most_common(1)[0][0],
            'agreement_count': most_common_count,
            'total_agents': len(agent_decisions)
        }
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate similarity between two texts
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score (0-1)
        """
        # Simple word overlap similarity
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 and not words2:
            return 1.0
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0