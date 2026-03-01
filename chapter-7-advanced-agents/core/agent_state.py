"""
Agent state management for maintaining context across interactions.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
import json


@dataclass
class AgentState:
    """Maintains agent state across execution"""
    
    agent_id: str
    task: str
    status: str = "initialized"  # initialized, running, completed, failed
    current_goal: Optional[str] = None
    subgoals: List[str] = field(default_factory=list)
    completed_subgoals: List[str] = field(default_factory=list)
    variables: Dict[str, Any] = field(default_factory=dict)
    observations: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def update(self, **kwargs):
        """Update state fields"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        self.updated_at = datetime.now()
    
    def add_observation(self, observation: Dict[str, Any]):
        """Add observation to state"""
        self.observations.append({
            **observation,
            'timestamp': datetime.now().isoformat()
        })
        self.updated_at = datetime.now()
    
    def set_variable(self, key: str, value: Any):
        """Set state variable"""
        self.variables[key] = value
        self.updated_at = datetime.now()
    
    def get_variable(self, key: str, default: Any = None) -> Any:
        """Get state variable"""
        return self.variables.get(key, default)
    
    def complete_subgoal(self, subgoal: str):
        """Mark subgoal as completed"""
        if subgoal in self.subgoals and subgoal not in self.completed_subgoals:
            self.completed_subgoals.append(subgoal)
            self.updated_at = datetime.now()
    
    def is_complete(self) -> bool:
        """Check if all subgoals completed"""
        return len(self.completed_subgoals) == len(self.subgoals)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'agent_id': self.agent_id,
            'task': self.task,
            'status': self.status,
            'current_goal': self.current_goal,
            'subgoals': self.subgoals,
            'completed_subgoals': self.completed_subgoals,
            'variables': self.variables,
            'observations': self.observations,
            'metadata': self.metadata,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AgentState':
        """Create from dictionary"""
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        data['updated_at'] = datetime.fromisoformat(data['updated_at'])
        return cls(**data)
    
    def save(self, filepath: str):
        """Save state to file"""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> 'AgentState':
        """Load state from file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)


class StateManager:
    """Manages multiple agent states"""
    
    def __init__(self):
        self.states: Dict[str, AgentState] = {}
    
    def create_state(self, agent_id: str, task: str, **kwargs) -> AgentState:
        """Create new agent state"""
        state = AgentState(agent_id=agent_id, task=task, **kwargs)
        self.states[agent_id] = state
        return state
    
    def get_state(self, agent_id: str) -> Optional[AgentState]:
        """Get agent state"""
        return self.states.get(agent_id)
    
    def update_state(self, agent_id: str, **kwargs):
        """Update agent state"""
        if agent_id in self.states:
            self.states[agent_id].update(**kwargs)
    
    def delete_state(self, agent_id: str):
        """Delete agent state"""
        if agent_id in self.states:
            del self.states[agent_id]
    
    def list_active_states(self) -> List[AgentState]:
        """List all active agent states"""
        return [s for s in self.states.values() if s.status == "running"]