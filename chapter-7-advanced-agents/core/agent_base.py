"""
Base agent classes and interfaces for all agent implementations.
"""

from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from abc import ABC, abstractmethod
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class AgentAction:
    """Represents an action taken by an agent"""
    action_type: str
    tool_name: Optional[str] = None
    tool_input: Optional[Dict[str, Any]] = None
    reasoning: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class AgentObservation:
    """Represents an observation from executing an action"""
    result: Any
    success: bool
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class AgentStep:
    """Represents a single step in agent execution"""
    step_number: int
    thought: str
    action: AgentAction
    observation: Optional[AgentObservation] = None
    timestamp: datetime = field(default_factory=datetime.now)


class Agent(ABC):
    """Base class for all agents"""
    
    def __init__(self, name: str, llm_client, tools: Optional[List] = None):
        """
        Initialize agent
        
        Args:
            name: Agent name
            llm_client: LLM client (Anthropic, OpenAI, etc.)
            tools: List of available tools
        """
        self.name = name
        self.llm = llm_client
        self.tools = tools or []
        self.execution_history: List[AgentStep] = []
        self.current_step = 0
    
    @abstractmethod
    def think(self, input_text: str, context: Optional[Dict[str, Any]] = None) -> str:
        """
        Generate thought/reasoning for next action
        
        Args:
            input_text: User input or current state
            context: Additional context
            
        Returns:
            Thought text
        """
        pass
    
    @abstractmethod
    def act(self, thought: str) -> AgentAction:
        """
        Determine action based on thought
        
        Args:
            thought: Generated thought
            
        Returns:
            AgentAction to execute
        """
        pass
    
    @abstractmethod
    def observe(self, action: AgentAction) -> AgentObservation:
        """
        Execute action and observe result
        
        Args:
            action: Action to execute
            
        Returns:
            Observation from action
        """
        pass
    
    def run(self, task: str, max_steps: int = 10) -> Dict[str, Any]:
        """
        Execute agent on task
        
        Args:
            task: Task description
            max_steps: Maximum steps to execute
            
        Returns:
            Final result
        """
        logger.info(f"Agent {self.name} starting task: {task}")
        
        current_input = task
        
        for step in range(max_steps):
            self.current_step = step + 1
            
            # Think
            thought = self.think(current_input)
            logger.info(f"Step {self.current_step} - Thought: {thought}")
            
            # Act
            action = self.act(thought)
            logger.info(f"Step {self.current_step} - Action: {action.action_type}")
            
            # Check if done
            if action.action_type == "finish":
                logger.info(f"Task completed in {self.current_step} steps")
                return {
                    'success': True,
                    'result': action.tool_input.get('final_answer'),
                    'steps': self.execution_history,
                    'total_steps': self.current_step
                }
            
            # Observe
            observation = self.observe(action)
            logger.info(f"Step {self.current_step} - Observation: {observation.result}")
            
            # Record step
            agent_step = AgentStep(
                step_number=self.current_step,
                thought=thought,
                action=action,
                observation=observation
            )
            self.execution_history.append(agent_step)
            
            # Update input for next iteration
            current_input = self._format_next_input(thought, action, observation)
        
        logger.warning(f"Reached max steps ({max_steps}) without completion")
        return {
            'success': False,
            'result': 'Max steps reached without completion',
            'steps': self.execution_history,
            'total_steps': self.current_step
        }
    
    def _format_next_input(self, thought: str, action: AgentAction, 
                          observation: AgentObservation) -> str:
        """Format input for next iteration"""
        return f"""Previous thought: {thought}
Action taken: {action.action_type}
Observation: {observation.result}

What should I do next?"""
    
    def get_available_tools_description(self) -> str:
        """Get formatted description of available tools"""
        if not self.tools:
            return "No tools available."
        
        descriptions = []
        for tool in self.tools:
            descriptions.append(f"- {tool.name}: {tool.description}")
        
        return "\n".join(descriptions)
    
    def reset(self):
        """Reset agent state"""
        self.execution_history = []
        self.current_step = 0