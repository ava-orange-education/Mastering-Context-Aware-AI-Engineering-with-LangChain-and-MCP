"""
Task decomposition and delegation strategies.
"""

from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)


class TaskDecomposer:
    """Decomposes complex tasks into subtasks"""
    
    def __init__(self, llm_client):
        """
        Initialize task decomposer
        
        Args:
            llm_client: LLM client for task analysis
        """
        self.llm = llm_client
    
    def decompose_task(self, task: str, num_subtasks: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Decompose task into subtasks
        
        Args:
            task: Main task description
            num_subtasks: Target number of subtasks (optional)
            
        Returns:
            List of subtask definitions
        """
        prompt = f"""Decompose this task into clear, independent subtasks:

Task: {task}

For each subtask, provide:
1. Subtask description
2. Required capabilities/skills
3. Expected output
4. Dependencies on other subtasks (if any)

{f'Aim for approximately {num_subtasks} subtasks.' if num_subtasks else ''}

Return as a structured list."""
        
        messages = [{"role": "user", "content": prompt}]
        
        response = self.llm.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=2048,
            messages=messages
        )
        
        decomposition_text = response.content[0].text
        
        # Parse subtasks (simplified - in production, use structured output)
        subtasks = self._parse_subtasks(decomposition_text)
        
        logger.info(f"Decomposed task into {len(subtasks)} subtasks")
        
        return subtasks
    
    def _parse_subtasks(self, text: str) -> List[Dict[str, Any]]:
        """Parse subtasks from LLM output"""
        # Simplified parsing - in production, use structured output or better parsing
        import re
        
        subtasks = []
        
        # Split by numbered items
        items = re.split(r'\n\d+\.', text)
        
        for i, item in enumerate(items[1:], 1):  # Skip first empty item
            subtasks.append({
                'id': f"subtask_{i}",
                'description': item.strip()[:200],  # First 200 chars
                'capabilities': ['general'],  # Would extract from text
                'dependencies': []
            })
        
        return subtasks
    
    def create_dependency_graph(self, subtasks: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """
        Create dependency graph for subtasks
        
        Args:
            subtasks: List of subtasks
            
        Returns:
            Dependency graph (subtask_id -> list of dependency ids)
        """
        graph = {}
        
        for subtask in subtasks:
            graph[subtask['id']] = subtask.get('dependencies', [])
        
        return graph
    
    def get_execution_order(self, subtasks: List[Dict[str, Any]]) -> List[str]:
        """
        Get execution order respecting dependencies (topological sort)
        
        Args:
            subtasks: List of subtasks
            
        Returns:
            Ordered list of subtask IDs
        """
        graph = self.create_dependency_graph(subtasks)
        
        # Simple topological sort
        visited = set()
        order = []
        
        def visit(node):
            if node in visited:
                return
            
            visited.add(node)
            
            for dep in graph.get(node, []):
                visit(dep)
            
            order.append(node)
        
        for subtask in subtasks:
            visit(subtask['id'])
        
        return order


class TaskDelegator:
    """Delegates subtasks to appropriate agents"""
    
    def __init__(self, agent_team):
        """
        Initialize task delegator
        
        Args:
            agent_team: AgentTeam instance
        """
        self.team = agent_team
    
    def delegate_subtasks(self, subtasks: List[Dict[str, Any]]) -> Dict[str, str]:
        """
        Delegate subtasks to agents
        
        Args:
            subtasks: List of subtasks
            
        Returns:
            Mapping of subtask_id -> agent_name
        """
        assignments = {}
        
        for subtask in subtasks:
            required_caps = subtask.get('capabilities', [])
            
            # Find best agent for this subtask
            agent = self.team.select_agent_for_task(
                task=subtask['description'],
                required_capabilities=required_caps
            )
            
            if agent:
                assignments[subtask['id']] = agent.name
                logger.info(f"Assigned {subtask['id']} to {agent.name}")
            else:
                logger.warning(f"No agent found for {subtask['id']}")
        
        return assignments
    
    def execute_delegated_tasks(self, subtasks: List[Dict[str, Any]], 
                               assignments: Dict[str, str],
                               execution_order: List[str]) -> Dict[str, Any]:
        """
        Execute delegated tasks in order
        
        Args:
            subtasks: List of subtasks
            assignments: Subtask assignments
            execution_order: Order of execution
            
        Returns:
            Execution results
        """
        results = {}
        
        for subtask_id in execution_order:
            # Find subtask
            subtask = next((s for s in subtasks if s['id'] == subtask_id), None)
            
            if not subtask:
                logger.warning(f"Subtask {subtask_id} not found")
                continue
            
            # Get assigned agent
            agent_name = assignments.get(subtask_id)
            
            if not agent_name:
                logger.warning(f"No agent assigned to {subtask_id}")
                continue
            
            agent = self.team.get_agent(agent_name)
            
            if not agent:
                logger.error(f"Agent {agent_name} not found")
                continue
            
            # Execute subtask
            logger.info(f"Executing {subtask_id} with agent {agent_name}")
            
            try:
                result = agent.run(subtask['description'], max_steps=5)
                results[subtask_id] = {
                    'agent': agent_name,
                    'result': result,
                    'success': result.get('success', False)
                }
            except Exception as e:
                logger.error(f"Subtask {subtask_id} failed: {e}")
                results[subtask_id] = {
                    'agent': agent_name,
                    'error': str(e),
                    'success': False
                }
        
        return results