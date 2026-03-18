"""
Example 5: Multi-Agent Evaluation
Demonstrates evaluation of multi-agent coordination.
"""

import sys
sys.path.append('..')

from multi_agent_eval.coordination_metrics import CoordinationMetrics
from multi_agent_eval.consensus_evaluator import ConsensusEvaluator
from multi_agent_eval.interaction_tracker import InteractionTracker


def main():
    print("=" * 70)
    print("  MULTI-AGENT EVALUATION EXAMPLE")
    print("=" * 70 + "\n")
    
    # Initialize components
    coord_metrics = CoordinationMetrics()
    consensus_eval = ConsensusEvaluator()
    interaction_tracker = InteractionTracker()
    
    # Simulate multi-agent system
    agents = ["research_agent", "analysis_agent", "synthesis_agent"]
    
    print("1. RECORDING AGENT INTERACTIONS")
    print("-" * 70)
    
    # Simulate interaction sequence
    interactions = [
        ("research_agent", "analysis_agent", "request"),
        ("analysis_agent", "research_agent", "response"),
        ("research_agent", "synthesis_agent", "notification"),
        ("analysis_agent", "synthesis_agent", "request"),
        ("synthesis_agent", "analysis_agent", "response"),
        ("synthesis_agent", "research_agent", "notification"),
    ]
    
    for sender, receiver, msg_type in interactions:
        coord_metrics.record_interaction(sender, receiver, msg_type)
        interaction_tracker.log_interaction(
            sender=sender,
            receiver=receiver,
            interaction_type=msg_type,
            content=f"Message from {sender} to {receiver}"
        )
        print(f"  {sender} → {receiver} ({msg_type})")
    
    print(f"\n✓ Recorded {len(interactions)} interactions")
    
    # Calculate coordination metrics
    print("\n2. COORDINATION METRICS")
    print("-" * 70)
    
    coordination = coord_metrics.calculate_coordination_score(agents)
    
    print(f"Coordination Score: {coordination['coordination_score']:.3f}")
    print(f"Balance Score: {coordination['balance_score']:.3f}")
    print(f"Density Score: {coordination['density_score']:.3f}")
    print(f"Total Interactions: {coordination['interaction_count']}")
    
    print("\nInteraction Distribution:")
    for agent, count in coordination['interaction_distribution'].items():
        print(f"  {agent}: {count} interactions")
    
    # Response time analysis
    print("\n3. RESPONSE TIME ANALYSIS")
    print("-" * 70)
    
    response_times = coord_metrics.calculate_response_time()
    
    if response_times.get('total_responses', 0) > 0:
        print(f"Average Response Time: {response_times['average_response_time']:.3f}s")
        print(f"Min Response Time: {response_times['min_response_time']:.3f}s")
        print(f"Max Response Time: {response_times['max_response_time']:.3f}s")
    else:
        print("Response time metrics: Insufficient data")
    
    # Centrality analysis
    print("\n4. AGENT CENTRALITY")
    print("-" * 70)
    
    for agent in agents:
        centrality = coord_metrics.calculate_centrality(agent)
        print(f"\n{agent}:")
        print(f"  Degree Centrality: {centrality['degree_centrality']:.3f}")
        print(f"  Outgoing: {centrality['outgoing_interactions']}")
        print(f"  Incoming: {centrality['incoming_interactions']}")
    
    # Consensus evaluation
    print("\n5. CONSENSUS EVALUATION")
    print("-" * 70)
    
    # Simulate agent responses to same query
    agent_responses = {
        "research_agent": "Machine learning is a subset of AI focused on learning from data.",
        "analysis_agent": "Machine learning enables systems to learn from data without explicit programming.",
        "synthesis_agent": "ML is an AI approach where systems improve through experience with data."
    }
    
    consensus = consensus_eval.evaluate_consensus(agent_responses)
    
    print(f"Consensus Score: {consensus['consensus_score']:.3f}")
    print(f"Agreement Level: {consensus['agreement_level']}")
    print(f"Number of Agents: {consensus['num_agents']}")
    
    # Find outliers
    outliers = consensus_eval.find_outliers(agent_responses)
    
    if outliers:
        print(f"\nOutlier Agents: {', '.join(outliers)}")
    else:
        print("\nNo outlier agents detected")
    
    # Decision quality
    print("\n6. DECISION QUALITY")
    print("-" * 70)
    
    agent_decisions = {
        "research_agent": "option_a",
        "analysis_agent": "option_a",
        "synthesis_agent": "option_b"
    }
    
    decision_quality = consensus_eval.evaluate_decision_quality(
        agent_decisions,
        ground_truth="option_a"
    )
    
    print(f"Decision Quality: {decision_quality['decision_quality']:.3f}")
    print(f"Correct Agents: {decision_quality['correct_agents']}/{decision_quality['total_agents']}")
    print(f"Accuracy: {decision_quality['accuracy']:.1%}")
    
    # Interaction analysis
    print("\n7. INTERACTION ANALYSIS")
    print("-" * 70)
    
    for agent in agents:
        metrics = interaction_tracker.calculate_interaction_metrics(agent)
        print(f"\n{agent}:")
        print(f"  Total Interactions: {metrics['total_interactions']}")
        print(f"  Sent: {metrics['sent']}, Received: {metrics['received']}")
        
        if metrics['top_partners']:
            print(f"  Top Partner: {metrics['top_partners'][0]['agent']} ({metrics['top_partners'][0]['count']} interactions)")
    
    # Collaboration pattern
    print("\n8. COLLABORATION PATTERN")
    print("-" * 70)
    
    pattern = interaction_tracker.analyze_collaboration_pattern()
    
    print(f"Pattern Type: {pattern['pattern']}")
    print(f"Coverage: {pattern['coverage']:.1%}")
    print(f"Total Agents: {pattern['total_agents']}")
    print(f"Active Pairs: {pattern['active_pairs']}")
    
    if pattern['hub_agents']:
        print(f"Hub Agents: {', '.join(pattern['hub_agents'])}")
    
    print("\n" + "=" * 70)
    print("  MULTI-AGENT EVALUATION DEMONSTRATION COMPLETE")
    print("=" * 70)
    
    print("\n💡 Key Multi-Agent Metrics:")
    print("   • Coordination Score - How well agents work together")
    print("   • Consensus Score - Agreement level on outputs")
    print("   • Centrality - Agent importance in network")
    print("   • Response Time - Communication efficiency")
    print("   • Collaboration Pattern - Overall interaction structure")


if __name__ == "__main__":
    main()