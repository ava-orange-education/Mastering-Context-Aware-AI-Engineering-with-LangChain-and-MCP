"""
Base tool interface and tool execution framework.
"""

from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ToolParameter:
    """Defines a tool parameter"""
    name: str
    type: str  # "string", "number", "boolean", "object", "array"
    description: str
    required: bool = True
    default: Any = None


class Tool(ABC):
    """Base class for all tools"""
    
    def __init__(self, name: str, description: str):
        """
        Initialize tool
        
        Args:
            name: Tool name
            description: What the tool does
        """
        self.name = name
        self.description = description
        self.parameters: List[ToolParameter] = []
    
    @abstractmethod
    def execute(self, input_data: Dict[str, Any]) -> Any:
        """
        Execute the tool
        
        Args:
            input_data: Tool input parameters
            
        Returns:
            Tool execution result
        """
        pass
    
    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """Validate input parameters"""
        for param in self.parameters:
            if param.required and param.name not in input_data:
                raise ValueError(f"Missing required parameter: {param.name}")
        
        return True
    
    def get_schema(self) -> Dict[str, Any]:
        """Get tool schema for LLM function calling"""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    param.name: {
                        "type": param.type,
                        "description": param.description
                    }
                    for param in self.parameters
                },
                "required": [p.name for p in self.parameters if p.required]
            }
        }
    
    def __str__(self) -> str:
        return f"Tool({self.name}): {self.description}"


class FunctionTool(Tool):
    """Tool that wraps a Python function"""
    
    def __init__(self, name: str, description: str, function: Callable, 
                 parameters: List[ToolParameter]):
        """
        Initialize function tool
        
        Args:
            name: Tool name
            description: Tool description
            function: Python function to execute
            parameters: Function parameters
        """
        super().__init__(name, description)
        self.function = function
        self.parameters = parameters
    
    def execute(self, input_data: Dict[str, Any]) -> Any:
        """Execute the wrapped function"""
        self.validate_input(input_data)
        
        try:
            result = self.function(**input_data)
            return result
        except Exception as e:
            logger.error(f"Error executing {self.name}: {e}")
            raise


class ToolExecutor:
    """Executes tools with error handling and logging"""
    
    def __init__(self):
        self.execution_history: List[Dict[str, Any]] = []
    
    def execute(self, tool: Tool, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute tool with error handling
        
        Args:
            tool: Tool to execute
            input_data: Input parameters
            
        Returns:
            Execution result with metadata
        """
        import time
        
        start_time = time.time()
        
        try:
            result = tool.execute(input_data)
            execution_time = time.time() - start_time
            
            execution_record = {
                'tool': tool.name,
                'input': input_data,
                'result': result,
                'success': True,
                'execution_time': execution_time,
                'error': None
            }
            
            self.execution_history.append(execution_record)
            
            logger.info(f"Successfully executed {tool.name} in {execution_time:.2f}s")
            
            return execution_record
        
        except Exception as e:
            execution_time = time.time() - start_time
            
            execution_record = {
                'tool': tool.name,
                'input': input_data,
                'result': None,
                'success': False,
                'execution_time': execution_time,
                'error': str(e)
            }
            
            self.execution_history.append(execution_record)
            
            logger.error(f"Failed to execute {tool.name}: {e}")
            
            return execution_record
    
    def get_execution_stats(self) -> Dict[str, Any]:
        """Get statistics about tool executions"""
        if not self.execution_history:
            return {'total': 0, 'success': 0, 'failure': 0}
        
        total = len(self.execution_history)
        success = sum(1 for e in self.execution_history if e['success'])
        
        return {
            'total': total,
            'success': success,
            'failure': total - success,
            'success_rate': success / total,
            'avg_execution_time': sum(e['execution_time'] for e in self.execution_history) / total
        }