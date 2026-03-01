"""
Example 2: Planning agent that creates and executes multi-step plans.
"""

import sys
sys.path.append('..')

from patterns.planning_agent import PlanningAgent
from tools.search_tools import WebSearchTool
from tools.file_tools import WriteFileTool
from tools.tool_base import FunctionTool, ToolParameter
from anthropic import Anthropic
import os


def analyze_data(data: str) -> str:
    """Simple data analysis function"""
    word_count = len(data.split())
    char_count = len(data)
    return f"Analysis: {word_count} words, {char_count} characters"


def main():
    print("=== Planning Agent Example ===\n")
    
    # Initialize LLM client
    llm_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    
    # Create tools
    search_tool = WebSearchTool()
    
    write_tool = WriteFileTool(allowed_paths=["./output"])
    
    analyze_tool = FunctionTool(
        name="analyze",
        description="Analyze text data",
        function=analyze_data,
        parameters=[
            ToolParameter(name="data", type="string", description="Data to analyze")
        ]
    )
    
    # Create planning agent
    agent = PlanningAgent(
        name="PlanningAgent",
        llm_client=llm_client,
        tools=[search_tool, write_tool, analyze_tool]
    )
    
    # Complex task requiring planning
    task = """Research the top 3 programming languages in 2024, 
    analyze their popularity, and write a summary report to a file."""
    
    print(f"Task: {task}\n")
    print("=" * 60 + "\n")
    
    # Run with planning
    result = agent.run_with_planning(task, max_steps=15)
    
    print("\n" + "=" * 60)
    print("\nExecution Plan:")
    for i, step in enumerate(result.get('plan', []), 1):
        print(f"{i}. {step.get('description', 'N/A')}")
    
    print(f"\nSteps Completed: {result['steps_completed']}/{len(result.get('plan', []))}")
    print(f"Success: {result['success']}")
    
    if result['success']:
        print(f"\nFinal Result: {result['result']}")


if __name__ == "__main__":
    main()