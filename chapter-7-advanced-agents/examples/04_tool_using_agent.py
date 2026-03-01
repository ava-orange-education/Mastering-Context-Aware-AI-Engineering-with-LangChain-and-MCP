"""
Example 4: Agent demonstrating comprehensive tool usage.
"""

import sys
sys.path.append('..')

from patterns.react_agent import ReActAgent
from tools.search_tools import WebSearchTool
from tools.file_tools import ReadFileTool, WriteFileTool, ListDirectoryTool
from tools.api_tools import APICallTool
from tools.code_tools import PythonExecutorTool
from tools.tool_registry import ToolRegistry
from anthropic import Anthropic
import os


def main():
    print("=== Tool-Using Agent Example ===\n")
    
    # Initialize LLM client
    llm_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    
    # Create tool registry
    registry = ToolRegistry()
    
    # Register various tools
    search_tool = WebSearchTool()
    read_tool = ReadFileTool(allowed_paths=["./data", "./"])
    write_tool = WriteFileTool(allowed_paths=["./output"])
    list_tool = ListDirectoryTool(allowed_paths=["./data", "./output"])
    python_tool = PythonExecutorTool(allowed_imports=['math', 'json', 'datetime'])
    
    registry.register(search_tool, "search")
    registry.register(read_tool, "file")
    registry.register(write_tool, "file")
    registry.register(list_tool, "file")
    registry.register(python_tool, "code")
    
    # Create agent with all tools
    agent = ReActAgent(
        name="ToolExpertAgent",
        llm_client=llm_client,
        tools=registry.list_tools()
    )
    
    print("Available Tools:")
    print(registry.get_tools_description())
    print("\n" + "=" * 60 + "\n")
    
    # Task using multiple tools
    task = """Calculate the factorial of 5 using Python code, 
    then save the result to a file named 'factorial_result.txt'."""
    
    print(f"Task: {task}\n")
    print("=" * 60 + "\n")
    
    # Run agent
    result = agent.run(task, max_steps=10)
    
    print("\n" + "=" * 60)
    print("\nExecution Trace:")
    for i, step in enumerate(result.get('steps', []), 1):
        print(f"\nStep {i}:")
        print(f"  Thought: {step.thought[:100]}...")
        print(f"  Action: {step.action.action_type}")
        if step.observation:
            obs_text = str(step.observation.result)[:150]
            print(f"  Observation: {obs_text}...")
    
    print("\n" + "=" * 60)
    print(f"\nFinal Result: {result['result']}")
    print(f"Success: {result['success']}")
    print(f"Total Steps: {result['total_steps']}")
    
    # Show tool usage statistics
    print("\n" + "=" * 60)
    print("\nTool Usage:")
    tool_counts = {}
    for step in result.get('steps', []):
        tool_name = step.action.tool_name or step.action.action_type
        tool_counts[tool_name] = tool_counts.get(tool_name, 0) + 1
    
    for tool, count in tool_counts.items():
        print(f"  {tool}: {count} times")


if __name__ == "__main__":
    main()