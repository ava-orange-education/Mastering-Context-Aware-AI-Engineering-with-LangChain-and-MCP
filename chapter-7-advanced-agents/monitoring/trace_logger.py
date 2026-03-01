"""
Detailed execution tracing for debugging and analysis.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
import json
import logging

logger = logging.getLogger(__name__)


class ExecutionTrace:
    """Records detailed execution trace"""
    
    def __init__(self, agent_name: str, task: str):
        """
        Initialize execution trace
        
        Args:
            agent_name: Agent name
            task: Task being executed
        """
        self.agent_name = agent_name
        self.task = task
        self.trace_id = f"{agent_name}_{datetime.now().timestamp()}"
        self.start_time = datetime.now()
        self.end_time = None
        
        self.steps: List[Dict[str, Any]] = []
        self.tool_calls: List[Dict[str, Any]] = []
        self.llm_calls: List[Dict[str, Any]] = []
        self.metadata: Dict[str, Any] = {}
    
    def log_step(self, step_num: int, thought: str, action: str, observation: str):
        """
        Log execution step
        
        Args:
            step_num: Step number
            thought: Agent's thought
            action: Action taken
            observation: Observation from action
        """
        self.steps.append({
            'step_num': step_num,
            'thought': thought,
            'action': action,
            'observation': observation,
            'timestamp': datetime.now().isoformat()
        })
    
    def log_tool_call(self, tool_name: str, input_data: Dict, output: Any, success: bool, execution_time: float):
        """Log tool call"""
        self.tool_calls.append({
            'tool': tool_name,
            'input': input_data,
            'output': output,
            'success': success,
            'execution_time': execution_time,
            'timestamp': datetime.now().isoformat()
        })
    
    def log_llm_call(self, prompt: str, response: str, model: str, tokens_used: Optional[int] = None):
        """Log LLM call"""
        self.llm_calls.append({
            'prompt': prompt[:500],  # First 500 chars
            'response': response[:500],
            'model': model,
            'tokens_used': tokens_used,
            'timestamp': datetime.now().isoformat()
        })
    
    def finalize(self, result: Any, success: bool):
        """Finalize trace"""
        self.end_time = datetime.now()
        self.metadata['result'] = result
        self.metadata['success'] = success
        self.metadata['duration'] = (self.end_time - self.start_time).total_seconds()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert trace to dictionary"""
        return {
            'trace_id': self.trace_id,
            'agent_name': self.agent_name,
            'task': self.task,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'steps': self.steps,
            'tool_calls': self.tool_calls,
            'llm_calls': self.llm_calls,
            'metadata': self.metadata
        }
    
    def save(self, filepath: str):
        """Save trace to file"""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        
        logger.info(f"Saved trace to {filepath}")


class TraceLogger:
    """Manages execution traces"""
    
    def __init__(self, output_dir: str = "./traces"):
        """
        Initialize trace logger
        
        Args:
            output_dir: Directory for trace files
        """
        import os
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.active_traces: Dict[str, ExecutionTrace] = {}
        self.completed_traces: List[ExecutionTrace] = []
    
    def start_trace(self, agent_name: str, task: str) -> str:
        """
        Start new execution trace
        
        Args:
            agent_name: Agent name
            task: Task description
            
        Returns:
            Trace ID
        """
        trace = ExecutionTrace(agent_name, task)
        self.active_traces[trace.trace_id] = trace
        
        logger.info(f"Started trace: {trace.trace_id}")
        
        return trace.trace_id
    
    def get_trace(self, trace_id: str) -> Optional[ExecutionTrace]:
        """Get active trace"""
        return self.active_traces.get(trace_id)
    
    def finalize_trace(self, trace_id: str, result: Any, success: bool):
        """Finalize and save trace"""
        trace = self.active_traces.get(trace_id)
        
        if not trace:
            logger.warning(f"Trace {trace_id} not found")
            return
        
        trace.finalize(result, success)
        
        # Save to file
        filename = f"{trace.agent_name}_{trace.start_time.strftime('%Y%m%d_%H%M%S')}.json"
        filepath = f"{self.output_dir}/{filename}"
        trace.save(filepath)
        
        # Move to completed
        self.completed_traces.append(trace)
        del self.active_traces[trace_id]
        
        logger.info(f"Finalized trace: {trace_id}")
    
    def get_trace_summary(self, trace_id: str) -> Optional[Dict[str, Any]]:
        """Get summary of a trace"""
        # Check active traces
        trace = self.active_traces.get(trace_id)
        
        if not trace:
            # Check completed traces
            trace = next((t for t in self.completed_traces if t.trace_id == trace_id), None)
        
        if not trace:
            return None
        
        return {
            'trace_id': trace.trace_id,
            'agent': trace.agent_name,
            'task': trace.task,
            'num_steps': len(trace.steps),
            'num_tool_calls': len(trace.tool_calls),
            'num_llm_calls': len(trace.llm_calls),
            'duration': trace.metadata.get('duration'),
            'success': trace.metadata.get('success')
        }