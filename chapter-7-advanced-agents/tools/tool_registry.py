"""
Tool registry for discovering and managing available tools.
"""

from typing import Dict, Any, List, Optional, Type
from .tool_base import Tool
import logging

logger = logging.getLogger(__name__)


class ToolRegistry:
    """Central registry for all available tools"""
    
    def __init__(self):
        self.tools: Dict[str, Tool] = {}
        self.tool_categories: Dict[str, List[str]] = {}
    
    def register(self, tool: Tool, category: str = "general"):
        """
        Register a tool
        
        Args:
            tool: Tool instance to register
            category: Tool category
        """
        self.tools[tool.name] = tool
        
        if category not in self.tool_categories:
            self.tool_categories[category] = []
        
        self.tool_categories[category].append(tool.name)
        
        logger.info(f"Registered tool: {tool.name} (category: {category})")
    
    def get_tool(self, name: str) -> Optional[Tool]:
        """Get tool by name"""
        return self.tools.get(name)
    
    def list_tools(self, category: Optional[str] = None) -> List[Tool]:
        """
        List all tools or tools in specific category
        
        Args:
            category: Optional category filter
            
        Returns:
            List of tools
        """
        if category:
            tool_names = self.tool_categories.get(category, [])
            return [self.tools[name] for name in tool_names]
        
        return list(self.tools.values())
    
    def get_tools_description(self, category: Optional[str] = None) -> str:
        """Get formatted description of tools"""
        tools = self.list_tools(category)
        
        if not tools:
            return "No tools available."
        
        descriptions = []
        for tool in tools:
            descriptions.append(f"- {tool.name}: {tool.description}")
        
        return "\n".join(descriptions)
    
    def get_tools_schemas(self, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get JSON schemas for all tools"""
        tools = self.list_tools(category)
        return [tool.get_schema() for tool in tools]
    
    def search_tools(self, query: str) -> List[Tool]:
        """Search tools by name or description"""
        query_lower = query.lower()
        
        matching_tools = []
        for tool in self.tools.values():
            if query_lower in tool.name.lower() or query_lower in tool.description.lower():
                matching_tools.append(tool)
        
        return matching_tools
    
    def unregister(self, tool_name: str):
        """Unregister a tool"""
        if tool_name in self.tools:
            # Remove from main registry
            del self.tools[tool_name]
            
            # Remove from categories
            for category, tools in self.tool_categories.items():
                if tool_name in tools:
                    tools.remove(tool_name)
            
            logger.info(f"Unregistered tool: {tool_name}")
    
    def clear(self):
        """Clear all registered tools"""
        self.tools.clear()
        self.tool_categories.clear()
        logger.info("Cleared all tools from registry")


# Global tool registry instance
global_tool_registry = ToolRegistry()


def register_tool(tool: Tool, category: str = "general"):
    """Register tool in global registry"""
    global_tool_registry.register(tool, category)


def get_tool(name: str) -> Optional[Tool]:
    """Get tool from global registry"""
    return global_tool_registry.get_tool(name)


def list_tools(category: Optional[str] = None) -> List[Tool]:
    """List tools from global registry"""
    return global_tool_registry.list_tools(category)