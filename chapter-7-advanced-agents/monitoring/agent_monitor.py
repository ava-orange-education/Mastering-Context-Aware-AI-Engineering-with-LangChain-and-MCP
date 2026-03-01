"""
Agent behavior monitoring and tracking.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class AgentMonitor:
    """Monitor agent behavior and performance"""
    
    def __init__(self, agent_name: str):
        """
        Initialize agent monitor
        
        Args:
            agent_name: Name of agent being monitored
        """
        self.agent_name = agent_name
        self.executions: List[Dict[str, Any]] = []
        self.tool_usage: Dict[str, int] = {}
        self.errors: List[Dict[str, Any]] = []
        self.metrics = {
            'total_tasks': 0,
            'successful_tasks': 0,
            'failed_tasks': 0,
            'total_steps': 0,
            'total_tool_calls': 0,
            'total_errors': 0
        }
    
    def log_execution(self, task: str, result: Dict[str, Any], execution_time: float):
        """
        Log agent execution
        
        Args:
            task: Task description
            result: Execution result
            execution_time: Time taken in seconds
        """
        execution_record = {
            'timestamp': datetime.now().isoformat(),
            'task': task,
            'result': result,
            'execution_time': execution_time,
            'success': result.get('success', False),
            'steps': result.get('total_steps', 0)
        }
        
        self.executions.append(execution_record)
        
        # Update metrics
        self.metrics['total_tasks'] += 1
        self.metrics['total_steps'] += result.get('total_steps', 0)
        
        if result.get('success'):
            self.metrics['successful_tasks'] += 1
        else:
            self.metrics['failed_tasks'] += 1
        
        logger.info(f"Logged execution for {self.agent_name}: {task[:50]}...")
    
    def log_tool_usage(self, tool_name: str, success: bool):
        """
        Log tool usage
        
        Args:
            tool_name: Name of tool used
            success: Whether tool call succeeded
        """
        if tool_name not in self.tool_usage:
            self.tool_usage[tool_name] = 0
        
        self.tool_usage[tool_name] += 1
        self.metrics['total_tool_calls'] += 1
        
        if not success:
            self.metrics['total_errors'] += 1
    
    def log_error(self, error_type: str, error_message: str, context: Optional[Dict] = None):
        """
        Log error
        
        Args:
            error_type: Type of error
            error_message: Error message
            context: Additional context
        """
        error_record = {
            'timestamp': datetime.now().isoformat(),
            'type': error_type,
            'message': error_message,
            'context': context or {}
        }
        
        self.errors.append(error_record)
        self.metrics['total_errors'] += 1
        
        logger.error(f"Agent {self.agent_name} error: {error_type} - {error_message}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        success_rate = (self.metrics['successful_tasks'] / self.metrics['total_tasks']
                       if self.metrics['total_tasks'] > 0 else 0)
        
        avg_steps = (self.metrics['total_steps'] / self.metrics['total_tasks']
                    if self.metrics['total_tasks'] > 0 else 0)
        
        return {
            **self.metrics,
            'success_rate': success_rate,
            'avg_steps_per_task': avg_steps,
            'most_used_tools': self._get_top_tools(5),
            'recent_errors': len([e for e in self.errors if self._is_recent(e['timestamp'])])
        }
    
    def _get_top_tools(self, n: int) -> List[Dict[str, Any]]:
        """Get top N most used tools"""
        sorted_tools = sorted(self.tool_usage.items(), key=lambda x: x[1], reverse=True)
        return [{'tool': tool, 'count': count} for tool, count in sorted_tools[:n]]
    
    def _is_recent(self, timestamp_str: str, hours: int = 24) -> bool:
        """Check if timestamp is recent"""
        from datetime import timedelta
        
        timestamp = datetime.fromisoformat(timestamp_str)
        cutoff = datetime.now() - timedelta(hours=hours)
        return timestamp > cutoff
    
    def get_execution_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get execution history"""
        if limit:
            return self.executions[-limit:]
        return self.executions
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get error summary"""
        error_types = {}
        
        for error in self.errors:
            error_type = error['type']
            if error_type not in error_types:
                error_types[error_type] = 0
            error_types[error_type] += 1
        
        return {
            'total_errors': len(self.errors),
            'error_types': error_types,
            'recent_errors': [e for e in self.errors if self._is_recent(e['timestamp'])]
        }
    
    def reset_metrics(self):
        """Reset all metrics"""
        self.executions = []
        self.tool_usage = {}
        self.errors = []
        self.metrics = {
            'total_tasks': 0,
            'successful_tasks': 0,
            'failed_tasks': 0,
            'total_steps': 0,
            'total_tool_calls': 0,
            'total_errors': 0
        }
        logger.info(f"Reset metrics for {self.agent_name}")


class MultiAgentMonitor:
    """Monitor multiple agents"""
    
    def __init__(self):
        self.agent_monitors: Dict[str, AgentMonitor] = {}
    
    def register_agent(self, agent_name: str) -> AgentMonitor:
        """Register agent for monitoring"""
        if agent_name not in self.agent_monitors:
            self.agent_monitors[agent_name] = AgentMonitor(agent_name)
        
        return self.agent_monitors[agent_name]
    
    def get_monitor(self, agent_name: str) -> Optional[AgentMonitor]:
        """Get monitor for agent"""
        return self.agent_monitors.get(agent_name)
    
    def get_aggregate_metrics(self) -> Dict[str, Any]:
        """Get aggregate metrics across all agents"""
        total_tasks = sum(m.metrics['total_tasks'] for m in self.agent_monitors.values())
        total_successful = sum(m.metrics['successful_tasks'] for m in self.agent_monitors.values())
        total_errors = sum(m.metrics['total_errors'] for m in self.agent_monitors.values())
        
        return {
            'num_agents': len(self.agent_monitors),
            'total_tasks': total_tasks,
            'total_successful': total_successful,
            'total_failed': total_tasks - total_successful,
            'overall_success_rate': total_successful / total_tasks if total_tasks > 0 else 0,
            'total_errors': total_errors,
            'agent_metrics': {
                name: monitor.get_metrics()
                for name, monitor in self.agent_monitors.items()
            }
        }