"""
Custom trace logging for agent execution.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
from dataclasses import dataclass, field
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TraceStep:
    """Single step in execution trace"""
    step_id: str
    step_type: str  # thought, action, observation, tool_call
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    duration_ms: Optional[float] = None


@dataclass
class ExecutionTrace:
    """Complete execution trace"""
    trace_id: str
    agent_name: str
    query: str
    response: Optional[str] = None
    steps: List[TraceStep] = field(default_factory=list)
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    total_duration_ms: Optional[float] = None
    status: str = "running"  # running, completed, failed
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class TraceLogger:
    """Log and manage execution traces"""
    
    def __init__(self, log_file: Optional[str] = None):
        """
        Initialize trace logger
        
        Args:
            log_file: Optional file path for trace logs
        """
        self.log_file = log_file
        self.traces: Dict[str, ExecutionTrace] = {}
        self.trace_counter = 0
    
    def start_trace(self, agent_name: str, query: str,
                   metadata: Optional[Dict] = None) -> str:
        """
        Start new execution trace
        
        Args:
            agent_name: Agent name
            query: Input query
            metadata: Additional metadata
            
        Returns:
            Trace ID
        """
        self.trace_counter += 1
        trace_id = f"trace_{self.trace_counter}_{int(datetime.now().timestamp())}"
        
        trace = ExecutionTrace(
            trace_id=trace_id,
            agent_name=agent_name,
            query=query,
            metadata=metadata or {}
        )
        
        self.traces[trace_id] = trace
        
        logger.info(f"Trace started: {trace_id} for agent {agent_name}")
        
        return trace_id
    
    def log_step(self, trace_id: str, step_type: str, content: str,
                metadata: Optional[Dict] = None, duration_ms: Optional[float] = None):
        """
        Log execution step
        
        Args:
            trace_id: Trace ID
            step_type: Type of step
            content: Step content
            metadata: Additional metadata
            duration_ms: Step duration in milliseconds
        """
        trace = self.traces.get(trace_id)
        
        if not trace:
            logger.warning(f"Trace not found: {trace_id}")
            return
        
        step_id = f"{trace_id}_step_{len(trace.steps) + 1}"
        
        step = TraceStep(
            step_id=step_id,
            step_type=step_type,
            content=content,
            metadata=metadata or {},
            duration_ms=duration_ms
        )
        
        trace.steps.append(step)
        
        logger.debug(f"Step logged: {step_type} for trace {trace_id}")
    
    def end_trace(self, trace_id: str, response: Optional[str] = None,
                 status: str = "completed", error: Optional[str] = None):
        """
        End execution trace
        
        Args:
            trace_id: Trace ID
            response: Final response
            status: Final status
            error: Error message if failed
        """
        trace = self.traces.get(trace_id)
        
        if not trace:
            logger.warning(f"Trace not found: {trace_id}")
            return
        
        trace.end_time = datetime.now()
        trace.response = response
        trace.status = status
        trace.error = error
        
        # Calculate total duration
        if trace.start_time and trace.end_time:
            duration = (trace.end_time - trace.start_time).total_seconds() * 1000
            trace.total_duration_ms = duration
        
        logger.info(f"Trace ended: {trace_id} ({status})")
        
        # Save to file if configured
        if self.log_file:
            self._save_trace_to_file(trace)
    
    def get_trace(self, trace_id: str) -> Optional[ExecutionTrace]:
        """Get trace by ID"""
        return self.traces.get(trace_id)
    
    def get_recent_traces(self, count: int = 10) -> List[ExecutionTrace]:
        """Get most recent traces"""
        all_traces = list(self.traces.values())
        all_traces.sort(key=lambda t: t.start_time, reverse=True)
        return all_traces[:count]
    
    def get_traces_by_agent(self, agent_name: str) -> List[ExecutionTrace]:
        """Get all traces for specific agent"""
        return [t for t in self.traces.values() if t.agent_name == agent_name]
    
    def get_trace_summary(self, trace_id: str) -> Optional[Dict[str, Any]]:
        """
        Get trace summary
        
        Args:
            trace_id: Trace ID
            
        Returns:
            Summary dictionary
        """
        trace = self.traces.get(trace_id)
        
        if not trace:
            return None
        
        return {
            'trace_id': trace.trace_id,
            'agent_name': trace.agent_name,
            'query': trace.query,
            'response': trace.response,
            'status': trace.status,
            'step_count': len(trace.steps),
            'total_duration_ms': trace.total_duration_ms,
            'start_time': trace.start_time.isoformat(),
            'end_time': trace.end_time.isoformat() if trace.end_time else None
        }
    
    def _save_trace_to_file(self, trace: ExecutionTrace):
        """Save trace to log file"""
        if not self.log_file:
            return
        
        trace_dict = {
            'trace_id': trace.trace_id,
            'agent_name': trace.agent_name,
            'query': trace.query,
            'response': trace.response,
            'status': trace.status,
            'error': trace.error,
            'start_time': trace.start_time.isoformat(),
            'end_time': trace.end_time.isoformat() if trace.end_time else None,
            'total_duration_ms': trace.total_duration_ms,
            'steps': [
                {
                    'step_id': s.step_id,
                    'step_type': s.step_type,
                    'content': s.content,
                    'timestamp': s.timestamp.isoformat(),
                    'duration_ms': s.duration_ms,
                    'metadata': s.metadata
                }
                for s in trace.steps
            ],
            'metadata': trace.metadata
        }
        
        try:
            with open(self.log_file, 'a') as f:
                f.write(json.dumps(trace_dict) + '\n')
        except Exception as e:
            logger.error(f"Failed to save trace to file: {e}")
    
    def export_traces(self, filepath: str, trace_ids: Optional[List[str]] = None):
        """
        Export traces to JSON file
        
        Args:
            filepath: Output file path
            trace_ids: Specific trace IDs to export (None for all)
        """
        traces_to_export = []
        
        if trace_ids:
            traces_to_export = [self.traces[tid] for tid in trace_ids if tid in self.traces]
        else:
            traces_to_export = list(self.traces.values())
        
        export_data = []
        
        for trace in traces_to_export:
            trace_dict = {
                'trace_id': trace.trace_id,
                'agent_name': trace.agent_name,
                'query': trace.query,
                'response': trace.response,
                'status': trace.status,
                'start_time': trace.start_time.isoformat(),
                'end_time': trace.end_time.isoformat() if trace.end_time else None,
                'total_duration_ms': trace.total_duration_ms,
                'step_count': len(trace.steps),
                'steps': [
                    {
                        'step_type': s.step_type,
                        'content': s.content[:100],  # Truncate for export
                        'duration_ms': s.duration_ms
                    }
                    for s in trace.steps
                ]
            }
            export_data.append(trace_dict)
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Exported {len(export_data)} traces to {filepath}")