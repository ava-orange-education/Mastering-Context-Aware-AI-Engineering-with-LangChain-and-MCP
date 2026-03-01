"""
Example 1: Simple ReAct agent with basic tools.
"""

import sys
sys.path.append('..')

from core.agent_base import Agent
from patterns.react_agent import ReActAgent
from tools.search_tools import WebSearchTool
from tools.tool_base import FunctionTool, ToolParameter
from anthropic import Anthropic
import os


def calculator(operation: str, num1: float, num2: float) -> float:
    """Simple calculator function"""
    if operation == "add":
        return num1 + num2
    elif operation == "subtract":
        return num1 - num2
    elif operation == "multiply":
        return num1 * num2
    elif operation == "divide":
        return num1 / num2 if num2 != 0 else "Error: Division by zero"
    else:
        return "Error: Unknown operation"


def main():
    print("=== Simple ReAct Agent Example ===\n")
    
    # Initialize LLM client
    llm_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    
    # Create tools
    calc_tool = FunctionTool(
        name="calculator",
        description="Perform basic arithmetic operations",
        function=calculator,
        parameters=[
            ToolParameter(name="operation", type="string", 
                         description="Operation: add, subtract, multiply, divide"),
            ToolParameter(name="num1", type="number", description="First number"),
            ToolParameter(name="num2", type="number", description="Second number")
        ]
    )
    
    search_tool = WebSearchTool()
    
    # Create ReAct agent
    agent = ReActAgent(
        name="MathAssistant",
        llm_client=llm_client,
        tools=[calc_tool, search_tool]
    )
    
    # Test task
    task = "What is 15 multiplied by 23?"
    
    print(f"Task: {task}\n")
    print("=" * 60)
    
    # Run agent
    result = agent.run(task, max_steps=5)
    
    print("\n" + "=" * 60)
    print(f"\nResult: {result['result']}")
    print(f"Steps taken: {result['total_steps']}")
    print(f"Success: {result['success']}")


if __name__ == "__main__":
    main()