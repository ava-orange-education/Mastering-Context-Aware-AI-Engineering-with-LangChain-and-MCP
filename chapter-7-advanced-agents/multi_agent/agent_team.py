"""
Multi-agent coordination and team management.
"""

from typing import Dict, Any, List, Optional
from ..core.agent_base import Agent
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class AgentRole:
    """Defines an agent's role in a team"""
    name: str
    description: str
    capabilities: List[str]
    priority: int = 1  # Higher priority agents get tasks first


class AgentTeam:
    """Manages a team of agents working together"""
    
    def __init__(self, team_name: str):
        """
        Initialize agent team
        
        Args:
            team_name: Name of the team
        """
        self.team_name = team_name
        self.agents: Dict[str, Agent] = {}
        self.roles: Dict[str, AgentRole] = {}
        self.task_history: List[Dict[str, Any]] = []
    
    def add_agent(self, agent: Agent, role: AgentRole):
        """
        Add agent to team
        
        Args:
            agent: Agent instance
            role: Agent's role definition
        """
        self.agents[agent.name] = agent
        self.roles[agent.name] = role
        logger.info(f"Added agent {agent.name} to team {self.team_name} as {role.name}")
    
    def remove_agent(self, agent_name: str):
        """Remove agent from team"""
        if agent_name in self.agents:
            del self.agents[agent_name]
            del self.roles[agent_name]
            logger.info(f"Removed agent {agent_name} from team")
    
    def get_agent(self, agent_name: str) -> Optional[Agent]:
        """Get agent by name"""
        return self.agents.get(agent_name)
    
    def select_agent_for_task(self, task: str, required_capabilities: Optional[List[str]] = None) -> Optional[Agent]:
        """
        Select best agent for a task based on capabilities
        
        Args:
            task: Task description
            required_capabilities: Required capabilities
            
        Returns:
            Selected agent or None
        """
        if not required_capabilities:
            # Return highest priority agent
            sorted_agents = sorted(
                self.agents.items(),
                key=lambda x: self.roles[x[0]].priority,
                reverse=True
            )
            return sorted_agents[0][1] if sorted_agents else None
        
        # Find agents with required capabilities
        capable_agents = []
        
        for agent_name, agent in self.agents.items():
            role = self.roles[agent_name]
            
            # Check if agent has all required capabilities
            if all(cap in role.capabilities for cap in required_capabilities):
                capable_agents.append((agent, role.priority))
        
        if not capable_agents:
            logger.warning(f"No agent found with required capabilities: {required_capabilities}")
            return None
        
        # Return highest priority capable agent
        capable_agents.sort(key=lambda x: x[1], reverse=True)
        return capable_agents[0][0]
    
    def execute_task_collaborative(self, task: str, max_rounds: int = 3) -> Dict[str, Any]:
        """
        Execute task with multiple agents collaborating
        
        Args:
            task: Task description
            max_rounds: Maximum collaboration rounds
            
        Returns:
            Collaborative result
        """
        logger.info(f"Team {self.team_name} starting collaborative task: {task}")
        
        results = []
        current_context = task
        
        for round_num in range(max_rounds):
            round_results = []
            
            # Each agent contributes
            for agent_name, agent in self.agents.items():
                logger.info(f"Round {round_num + 1}: Agent {agent_name} contributing")
                
                try:
                    # Agent works on current context
                    agent_result = agent.run(current_context, max_steps=5)
                    
                    round_results.append({
                        'agent': agent_name,
                        'role': self.roles[agent_name].name,
                        'result': agent_result
                    })
                    
                except Exception as e:
                    logger.error(f"Agent {agent_name} failed: {e}")
                    round_results.append({
                        'agent': agent_name,
                        'error': str(e)
                    })
            
            results.append({
                'round': round_num + 1,
                'contributions': round_results
            })
            
            # Update context with results
            current_context = self._synthesize_results(round_results)
        
        # Store in history
        self.task_history.append({
            'task': task,
            'rounds': results,
            'final_context': current_context
        })
        
        return {
            'task': task,
            'team': self.team_name,
            'rounds': results,
            'final_result': current_context,
            'success': True
        }
    
    def _synthesize_results(self, round_results: List[Dict[str, Any]]) -> str:
        """Synthesize results from multiple agents"""
        synthesis = "Synthesis of agent contributions:\n\n"
        
        for result in round_results:
            if 'error' not in result:
                agent_output = result['result'].get('result', 'No output')
                synthesis += f"{result['agent']} ({result['role']}): {agent_output}\n\n"
        
        return synthesis
    
    def get_team_capabilities(self) -> List[str]:
        """Get all capabilities available in the team"""
        all_capabilities = set()
        
        for role in self.roles.values():
            all_capabilities.update(role.capabilities)
        
        return list(all_capabilities)
    
    def get_team_status(self) -> Dict[str, Any]:
        """Get team status information"""
        return {
            'team_name': self.team_name,
            'num_agents': len(self.agents),
            'agents': [
                {
                    'name': name,
                    'role': self.roles[name].name,
                    'capabilities': self.roles[name].capabilities
                }
                for name in self.agents.keys()
            ],
            'total_capabilities': len(self.get_team_capabilities()),
            'tasks_completed': len(self.task_history)
        }


class HierarchicalTeam(AgentTeam):
    """Team with hierarchical structure (manager + workers)"""
    
    def __init__(self, team_name: str, manager_agent: Agent):
        """
        Initialize hierarchical team
        
        Args:
            team_name: Team name
            manager_agent: Agent that coordinates others
        """
        super().__init__(team_name)
        self.manager = manager_agent
        self.add_agent(
            manager_agent,
            AgentRole(
                name="Manager",
                description="Coordinates team activities",
                capabilities=["planning", "coordination", "synthesis"],
                priority=10
            )
        )
    
    def execute_task_hierarchical(self, task: str) -> Dict[str, Any]:
        """
        Execute task with manager delegating to workers
        
        Args:
            task: Task description
            
        Returns:
            Task result
        """
        logger.info(f"Hierarchical team {self.team_name} starting task: {task}")
        
        # Manager creates plan
        plan_prompt = f"""As the team manager, create a plan to accomplish this task:

Task: {task}

Available team members:
{self._get_team_description()}

Create a plan that assigns subtasks to appropriate team members.
Return the plan as a list of subtasks with assigned agents."""
        
        manager_plan = self.manager.run(plan_prompt, max_steps=3)
        
        # Extract subtasks (simplified - in production, parse structured output)
        subtasks = self._extract_subtasks_from_plan(manager_plan)
        
        # Execute subtasks
        subtask_results = []
        
        for subtask in subtasks:
            agent_name = subtask.get('assigned_to')
            task_desc = subtask.get('task')
            
            agent = self.get_agent(agent_name)
            
            if not agent:
                logger.warning(f"Agent {agent_name} not found, skipping subtask")
                continue
            
            logger.info(f"Agent {agent_name} executing: {task_desc}")
            
            try:
                result = agent.run(task_desc, max_steps=5)
                subtask_results.append({
                    'subtask': task_desc,
                    'agent': agent_name,
                    'result': result
                })
            except Exception as e:
                logger.error(f"Subtask failed: {e}")
                subtask_results.append({
                    'subtask': task_desc,
                    'agent': agent_name,
                    'error': str(e)
                })
        
        # Manager synthesizes results
        synthesis_prompt = f"""Synthesize the following subtask results into a final answer:

Original task: {task}

Subtask results:
{self._format_subtask_results(subtask_results)}

Provide a comprehensive final answer."""
        
        final_result = self.manager.run(synthesis_prompt, max_steps=3)
        
        return {
            'task': task,
            'team': self.team_name,
            'plan': manager_plan,
            'subtask_results': subtask_results,
            'final_result': final_result,
            'success': True
        }
    
    def _get_team_description(self) -> str:
        """Get formatted team description"""
        descriptions = []
        
        for agent_name, agent in self.agents.items():
            if agent_name == self.manager.name:
                continue
            
            role = self.roles[agent_name]
            descriptions.append(f"- {agent_name}: {role.description} (capabilities: {', '.join(role.capabilities)})")
        
        return "\n".join(descriptions)
    
    def _extract_subtasks_from_plan(self, plan_result: Dict[str, Any]) -> List[Dict[str, str]]:
        """Extract subtasks from manager's plan (simplified)"""
        # In production, this would parse structured output
        # For now, return dummy subtasks
        return [
            {'task': 'Research the topic', 'assigned_to': list(self.agents.keys())[1] if len(self.agents) > 1 else self.manager.name},
            {'task': 'Analyze findings', 'assigned_to': list(self.agents.keys())[-1] if len(self.agents) > 1 else self.manager.name}
        ]
    
    def _format_subtask_results(self, results: List[Dict[str, Any]]) -> str:
        """Format subtask results for synthesis"""
        formatted = []
        
        for i, result in enumerate(results):
            formatted.append(f"{i+1}. {result['subtask']} (by {result['agent']}):")
            
            if 'error' in result:
                formatted.append(f"   Error: {result['error']}")
            else:
                formatted.append(f"   Result: {result['result'].get('result', 'No result')}")
        
        return "\n".join(formatted)