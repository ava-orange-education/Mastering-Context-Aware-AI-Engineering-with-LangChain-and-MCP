"""
Tools for code execution and manipulation.
"""

from typing import Dict, Any
from .tool_base import Tool, ToolParameter
import logging
import subprocess
import sys

logger = logging.getLogger(__name__)


class PythonExecutorTool(Tool):
    """Tool for executing Python code"""
    
    def __init__(self, allowed_imports: Optional[List[str]] = None, timeout: int = 10):
        super().__init__(
            name="python_executor",
            description="Execute Python code in a safe environment"
        )
        
        self.parameters = [
            ToolParameter(
                name="code",
                type="string",
                description="Python code to execute"
            ),
            ToolParameter(
                name="return_output",
                type="boolean",
                description="Whether to return stdout",
                required=False,
                default=True
            )
        ]
        
        self.allowed_imports = allowed_imports or ['math', 'json', 'datetime', 'random']
        self.timeout = timeout
    
    def execute(self, input_data: Dict[str, Any]) -> Any:
        """Execute Python code"""
        self.validate_input(input_data)
        
        code = input_data['code']
        return_output = input_data.get('return_output', True)
        
        logger.info(f"Executing Python code ({len(code)} chars)")
        
        # Security: Check for dangerous operations
        dangerous_keywords = ['import os', 'import sys', '__import__', 'eval', 'exec', 'open']
        
        for keyword in dangerous_keywords:
            if keyword in code:
                # Check if it's in allowed imports
                if keyword.startswith('import') and any(imp in code for imp in self.allowed_imports):
                    continue
                return {"error": f"Dangerous operation not allowed: {keyword}"}
        
        try:
            # Capture stdout
            from io import StringIO
            import contextlib
            
            stdout = StringIO()
            
            # Create restricted globals
            safe_globals = {
                '__builtins__': {
                    'print': print,
                    'len': len,
                    'range': range,
                    'str': str,
                    'int': int,
                    'float': float,
                    'list': list,
                    'dict': dict,
                    'set': set,
                    'tuple': tuple,
                    'abs': abs,
                    'min': min,
                    'max': max,
                    'sum': sum,
                }
            }
            
            # Allow specific imports
            for imp in self.allowed_imports:
                try:
                    safe_globals[imp] = __import__(imp)
                except:
                    pass
            
            # Execute code
            with contextlib.redirect_stdout(stdout):
                exec(code, safe_globals)
            
            output = stdout.getvalue()
            
            return {
                "success": True,
                "output": output if return_output else None,
                "error": None
            }
        
        except Exception as e:
            logger.error(f"Code execution failed: {e}")
            return {
                "success": False,
                "output": None,
                "error": str(e)
            }


class ShellCommandTool(Tool):
    """Tool for executing shell commands"""
    
    def __init__(self, allowed_commands: Optional[List[str]] = None):
        super().__init__(
            name="shell_command",
            description="Execute shell commands"
        )
        
        self.parameters = [
            ToolParameter(
                name="command",
                type="string",
                description="Shell command to execute"
            )
        ]
        
        # Whitelist of allowed commands for security
        self.allowed_commands = allowed_commands or ['ls', 'pwd', 'echo', 'cat', 'grep']
    
    def execute(self, input_data: Dict[str, Any]) -> Any:
        """Execute shell command"""
        self.validate_input(input_data)
        
        command = input_data['command']
        
        # Security: Check if command is allowed
        command_name = command.split()[0]
        
        if self.allowed_commands and command_name not in self.allowed_commands:
            return {"error": f"Command not allowed: {command_name}"}
        
        logger.info(f"Executing shell command: {command}")
        
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=10
            )
            
            return {
                "command": command,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode,
                "success": result.returncode == 0
            }
        
        except subprocess.TimeoutExpired:
            logger.error(f"Command timeout: {command}")
            return {"error": "Command timeout"}
        
        except Exception as e:
            logger.error(f"Command execution failed: {e}")
            return {"error": str(e)}