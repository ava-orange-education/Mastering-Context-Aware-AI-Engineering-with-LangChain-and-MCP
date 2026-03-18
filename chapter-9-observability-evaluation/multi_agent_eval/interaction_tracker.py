"""
Track and analyze interactions between agents.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from collections import defaultdict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InteractionTracker:
    """Track interactions between agents"""
    
    def __init__(self):
        """Initialize interaction tracker"""
        self.interactions: List[Dict[str, Any]] = []
        self.interaction_counter = 0
    
    def log_interaction(self, sender: str, receiver: str,
                       interaction_type: str, content: str,
                       metadata: Optional[Dict] = None) -> str:
        """
        Log agent interaction
        
        Args:
            sender: Sending agent
            receiver: Receiving agent
            interaction_type: Type of interaction (request, response, notification)
            content: Interaction content
            metadata: Additional metadata
            
        Returns:
            Interaction ID
        """
        self.interaction_counter += 1
        interaction_id = f"interaction_{self.interaction_counter}"
        
        interaction = {
            'interaction_id': interaction_id,
            'sender': sender,
            'receiver': receiver,
            'type': interaction_type,
            'content': content,
            'timestamp': datetime.now(),
            'metadata': metadata or {}
        }
        
        self.interactions.append(interaction)
        
        logger.debug(f"Interaction logged: {sender} -> {receiver} ({interaction_type})")
        
        return interaction_id
    
    def get_interaction_sequence(self, start_id: str) -> List[Dict[str, Any]]:
        """
        Get sequence of interactions starting from a given interaction
        
        Args:
            start_id: Starting interaction ID
            
        Returns:
            List of related interactions
        """
        # Find starting interaction
        start_interaction = None
        start_index = 0
        
        for idx, interaction in enumerate(self.interactions):
            if interaction['interaction_id'] == start_id:
                start_interaction = interaction
                start_index = idx
                break
        
        if not start_interaction:
            return []
        
        # Get subsequent related interactions
        sequence = [start_interaction]
        current_receiver = start_interaction['receiver']
        
        for interaction in self.interactions[start_index + 1:]:
            if interaction['sender'] == current_receiver:
                sequence.append(interaction)
                current_receiver = interaction['receiver']
            elif len(sequence) > 1:
                # Sequence broken
                break
        
        return sequence
    
    def get_interaction_chains(self) -> List[List[Dict[str, Any]]]:
        """
        Identify all interaction chains
        
        Returns:
            List of interaction chains
        """
        chains = []
        processed = set()
        
        for interaction in self.interactions:
            interaction_id = interaction['interaction_id']
            
            if interaction_id in processed:
                continue
            
            # Get chain starting from this interaction
            chain = self.get_interaction_sequence(interaction_id)
            
            if len(chain) > 1:
                chains.append(chain)
                
                # Mark all interactions in chain as processed
                for chain_interaction in chain:
                    processed.add(chain_interaction['interaction_id'])
        
        return chains
    
    def calculate_interaction_metrics(self, agent: str) -> Dict[str, Any]:
        """
        Calculate interaction metrics for specific agent
        
        Args:
            agent: Agent name
            
        Returns:
            Interaction metrics
        """
        # Count interactions
        sent = sum(1 for i in self.interactions if i['sender'] == agent)
        received = sum(1 for i in self.interactions if i['receiver'] == agent)
        total = sent + received
        
        # Interaction types
        types_sent = defaultdict(int)
        types_received = defaultdict(int)
        
        for interaction in self.interactions:
            if interaction['sender'] == agent:
                types_sent[interaction['type']] += 1
            if interaction['receiver'] == agent:
                types_received[interaction['type']] += 1
        
        # Most frequent partners
        partners = defaultdict(int)
        for interaction in self.interactions:
            if interaction['sender'] == agent:
                partners[interaction['receiver']] += 1
            if interaction['receiver'] == agent:
                partners[interaction['sender']] += 1
        
        top_partners = sorted(partners.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return {
            'total_interactions': total,
            'sent': sent,
            'received': received,
            'types_sent': dict(types_sent),
            'types_received': dict(types_received),
            'top_partners': [{'agent': agent, 'count': count} for agent, count in top_partners]
        }
    
    def get_interaction_frequency(self, time_window_minutes: int = 60) -> Dict[str, Any]:
        """
        Calculate interaction frequency
        
        Args:
            time_window_minutes: Time window for frequency calculation
            
        Returns:
            Frequency metrics
        """
        if not self.interactions:
            return {'interactions_per_minute': 0.0}
        
        cutoff = datetime.now() - timedelta(minutes=time_window_minutes)
        recent = [i for i in self.interactions if i['timestamp'] > cutoff]
        
        interactions_per_minute = len(recent) / time_window_minutes
        
        # Peak frequency (highest 5-minute window)
        if len(recent) >= 2:
            # Find 5-minute windows
            windows = []
            for i in range(len(recent)):
                window_start = recent[i]['timestamp']
                window_end = window_start + timedelta(minutes=5)
                
                count = sum(
                    1 for interaction in recent
                    if window_start <= interaction['timestamp'] < window_end
                )
                windows.append(count)
            
            peak_frequency = max(windows) / 5 if windows else 0
        else:
            peak_frequency = 0
        
        return {
            'interactions_per_minute': interactions_per_minute,
            'peak_frequency_per_minute': peak_frequency,
            'time_window_minutes': time_window_minutes,
            'total_recent_interactions': len(recent)
        }
    
    def analyze_collaboration_pattern(self) -> Dict[str, Any]:
        """
        Analyze overall collaboration patterns
        
        Returns:
            Collaboration analysis
        """
        if not self.interactions:
            return {'pattern': 'no_interactions'}
        
        # Get all agents
        agents = set()
        for interaction in self.interactions:
            agents.add(interaction['sender'])
            agents.add(interaction['receiver'])
        
        # Interaction distribution
        interaction_matrix = defaultdict(lambda: defaultdict(int))
        
        for interaction in self.interactions:
            sender = interaction['sender']
            receiver = interaction['receiver']
            interaction_matrix[sender][receiver] += 1
        
        # Analyze pattern
        total_pairs = len(agents) * (len(agents) - 1)
        active_pairs = sum(1 for sender in interaction_matrix.values() for count in sender.values() if count > 0)
        
        coverage = active_pairs / total_pairs if total_pairs > 0 else 0
        
        # Identify pattern type
        if coverage > 0.7:
            pattern = 'highly_collaborative'
        elif coverage > 0.4:
            pattern = 'moderately_collaborative'
        elif coverage > 0.2:
            pattern = 'selective_collaboration'
        else:
            pattern = 'isolated_agents'
        
        # Find hub agents (high interaction count)
        agent_totals = {}
        for agent in agents:
            total = sum(
                1 for i in self.interactions
                if i['sender'] == agent or i['receiver'] == agent
            )
            agent_totals[agent] = total
        
        avg_interactions = sum(agent_totals.values()) / len(agent_totals) if agent_totals else 0
        hubs = [
            agent for agent, count in agent_totals.items()
            if count > avg_interactions * 1.5
        ]
        
        return {
            'pattern': pattern,
            'coverage': coverage,
            'total_agents': len(agents),
            'active_pairs': active_pairs,
            'hub_agents': hubs,
            'avg_interactions_per_agent': avg_interactions
        }
    
    def get_interaction_timeline(self, agent: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get chronological timeline of interactions
        
        Args:
            agent: Optional agent to filter by
            
        Returns:
            Timeline of interactions
        """
        interactions = self.interactions
        
        if agent:
            interactions = [
                i for i in interactions
                if i['sender'] == agent or i['receiver'] == agent
            ]
        
        # Sort by timestamp
        timeline = sorted(interactions, key=lambda x: x['timestamp'])
        
        return [
            {
                'timestamp': i['timestamp'].isoformat(),
                'sender': i['sender'],
                'receiver': i['receiver'],
                'type': i['type']
            }
            for i in timeline
        ]   