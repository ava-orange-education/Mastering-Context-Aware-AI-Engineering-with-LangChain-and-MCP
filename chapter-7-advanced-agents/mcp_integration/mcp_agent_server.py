"""
MCP server implementation for agent tools and capabilities.
"""

from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)


class MCPAgentServer:
    """MCP server exposing agent capabilities"""
    
    def __init__(self, agent, server_name: str = "agent_server"):
        """
        Initialize MCP agent server
        
        Args:
            agent: Agent instance
            server_name: Server name
        """
        self.agent = agent
        self.server_name = server_name
        self.tools = agent.tools if hasattr(agent, 'tools') else []
    
    def get_server_info(self) -> Dict[str, Any]:
        """Get server information"""
        return {
            'name': self.server_name,
            'version': '1.0.0',
            'agent': self.agent.name,
            'capabilities': {
                'tools': True,
                'prompts': True,
                'resources': False
            }
        }
    
    def list_tools(self) -> List[Dict[str, Any]]:
        """List available tools"""
        return [
            {
                'name': tool.name,
                'description': tool.description,
                'inputSchema': tool.get_schema()
            }
            for tool in self.tools
        ]
    
    def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Call a tool
        
        Args:
            tool_name: Tool name
            arguments: Tool arguments
            
        Returns:
            Tool result
        """
        # Find tool
        tool = next((t for t in self.tools if t.name == tool_name), None)
        
        if not tool:
            return {
                'error': f"Tool not found: {tool_name}",
                'isError': True
            }
        
        try:
            result = tool.execute(arguments)
            
            return {
                'content': [
                    {
                        'type': 'text',
                        'text': str(result)
                    }
                ],
                'isError': False
            }
        
        except Exception as e:
            logger.error(f"Tool execution failed: {e}")
            return {
                'error': str(e),
                'isError': True
            }
    
    def get_prompt(self, prompt_name: str, arguments: Optional[Dict] = None) -> Dict[str, Any]:
        """Get a prompt template"""
        # Predefined prompts for the agent
        prompts = {
            'analyze': """Analyze the following and provide insights:

{input}

Provide a detailed analysis.""",
            
            'summarize': """Summarize the following content:

{input}

Provide a concise summary.""",
            
            'reason': """Reason through this problem step by step:

{problem}

Show your reasoning process."""
        }
        
        if prompt_name not in prompts:
            return {'error': f"Prompt not found: {prompt_name}"}
        
        template = prompts[prompt_name]
        
        # Fill in arguments if provided
        if arguments:
            try:
                template = template.format(**arguments)
            except KeyError as e:
                return {'error': f"Missing argument: {e}"}
        
        return {
            'messages': [
                {
                    'role': 'user',
                    'content': {
                        'type': 'text',
                        'text': template
                    }
                }
            ]
        }
    
    def handle_request(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle MCP request
        
        Args:
            method: Request method
            params: Request parameters
            
        Returns:
            Response
        """
        if method == 'tools/list':
            return {'tools': self.list_tools()}
        
        elif method == 'tools/call':
            return self.call_tool(
                params.get('name'),
                params.get('arguments', {})
            )
        
        elif method == 'prompts/get':
            return self.get_prompt(
                params.get('name'),
                params.get('arguments')
            )
        
        elif method == 'initialize':
            return self.get_server_info()
        
        else:
            return {'error': f"Unknown method: {method}"}