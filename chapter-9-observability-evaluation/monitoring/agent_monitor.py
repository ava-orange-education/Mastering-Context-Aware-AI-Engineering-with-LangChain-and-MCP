"""
Monitor agent behavior and performance.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from collections import defaultdict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AgentMonitor:
    """Monitor agent execution and behavior"""
    
    def __init__(self, agent_name: str):
        """
        Initialize agent monitor
        
        Args:
            agent_name: Name of agent to monitor
        """
        self.agent_name = agent_name
        self.execution_count = 0
        self.success_count = 0
        self.failure_count = 0
        self.total_latency = 0.0
        self.executions: List[Dict[str, Any]] = []
        self.errors: List[Dict[str, Any]] = []
    
    def record_execution(self, query: str, response: str, 
                        latency: float, success: bool = True,
                        metadata: Optional[Dict] = None):
        """
        Record agent execution
        
        Args:
            query: Input query
            response: Agent response
            latency: Execution time in seconds
            success: Whether execution succeeded
            metadata: Additional metadata
        """
        self.execution_count += 1
        
        if success:
            self.success_count += 1
        else:
            self.failure_count += 1
        
        self.total_latency += latency
        
        execution_record = {
            'execution_id': self.execution_count,
            'query': query,
            'response': response,
            'latency': latency,
            'success': success,
            'timestamp': datetime.now(),
            'metadata': metadata or {}
        }
        
        self.executions.append(execution_record)
        
        logger.debug(f"Execution recorded for {self.agent_name}: {success} in {latency:.3f}s")
    
    def record_error(self, query: str, error: str, error_type: str = "unknown"):
        """
        Record agent error
        
        Args:
            query: Query that caused error
            error: Error message
            error_type: Error type/category
        """
        error_record = {
            'query': query,
            'error': error,
            'error_type': error_type,
            'timestamp': datetime.now()
        }
        
        self.errors.append(error_record)
        
        logger.warning(f"Error recorded for {self.agent_name}: {error_type}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get agent metrics
        
        Returns:
            Metrics dictionary
        """
        avg_latency = self.total_latency / self.execution_count if self.execution_count > 0 else 0
        success_rate = self.success_count / self.execution_count if self.execution_count > 0 else 0
        
        return {
            'agent_name': self.agent_name,
            'total_executions': self.execution_count,
            'successful_executions': self.success_count,
            'failed_executions': self.failure_count,
            'success_rate': success_rate,
            'average_latency': avg_latency,
            'total_errors': len(self.errors)
        }
    
    def get_recent_executions(self, count: int = 10) -> List[Dict[str, Any]]:
        """Get most recent executions"""
        return self.executions[-count:]
    
    def get_recent_errors(self, count: int = 10) -> List[Dict[str, Any]]:
        """Get most recent errors"""
        return self.errors[-count:]
    
    def get_error_summary(self) -> Dict[str, int]:
        """Get error counts by type"""
        error_counts = defaultdict(int)
        
        for error in self.errors:
            error_counts[error['error_type']] += 1
        
        return dict(error_counts)
    
    def get_latency_percentiles(self) -> Dict[str, float]:
        """Calculate latency percentiles"""
        if not self.executions:
            return {}
        
        latencies = sorted([e['latency'] for e in self.executions])
        count = len(latencies)
        
        return {
            'p50': latencies[int(count * 0.5)],
            'p90': latencies[int(count * 0.9)],
            'p95': latencies[int(count * 0.95)],
            'p99': latencies[int(count * 0.99)],
            'min': latencies[0],
            'max': latencies[-1]
        }


class MultiAgentMonitor:
    """Monitor multiple agents"""
    
    def __init__(self):
        """Initialize multi-agent monitor"""
        self.agent_monitors: Dict[str, AgentMonitor] = {}
    
    def get_monitor(self, agent_name: str) -> AgentMonitor:
        """
        Get or create monitor for agent
        
        Args:
            agent_name: Agent name
            
        Returns:
            AgentMonitor instance
        """
        if agent_name not in self.agent_monitors:
            self.agent_monitors[agent_name] = AgentMonitor(agent_name)
        
        return self.agent_monitors[agent_name]
    
    def get_all_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get metrics for all agents"""
        return {
            name: monitor.get_metrics()
            for name, monitor in self.agent_monitors.items()
        }
    
    def get_summary(self) -> Dict[str, Any]:
        """Get overall summary across all agents"""
        total_executions = sum(m.execution_count for m in self.agent_monitors.values())
        total_successes = sum(m.success_count for m in self.agent_monitors.values())
        total_failures = sum(m.failure_count for m in self.agent_monitors.values())
        
        overall_success_rate = total_successes / total_executions if total_executions > 0 else 0
        
        return {
            'total_agents': len(self.agent_monitors),
            'total_executions': total_executions,
            'total_successes': total_successes,
            'total_failures': total_failures,
            'overall_success_rate': overall_success_rate,
            'agent_metrics': self.get_all_metrics()
        }