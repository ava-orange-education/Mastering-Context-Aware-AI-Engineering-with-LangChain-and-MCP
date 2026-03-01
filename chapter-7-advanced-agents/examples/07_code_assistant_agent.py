"""
Example 7: Code assistant agent for programming tasks.
"""

import sys
sys.path.append('..')

from patterns.react_agent import ReActAgent
from patterns.reflection_agent import SelfCritiqueAgent
from tools.code_tools import PythonExecutorTool
from tools.file_tools import ReadFileTool, WriteFileTool
from anthropic import Anthropic
import os


def main():
    print("=== Code Assistant Agent ===\n")
    
    # Initialize LLM client
    llm_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    
    # Create tools
    python_executor = PythonExecutorTool(
        allowed_imports=['math', 'json', 'datetime', 'random', 'statistics'],
        timeout=10
    )
    
    read_tool = ReadFileTool(allowed_paths=["./data", "./"])
    write_tool = WriteFileTool(allowed_paths=["./output"])
    
    # Create code assistant agent
    agent = ReActAgent(
        name="CodeAssistant",
        llm_client=llm_client,
        tools=[python_executor, read_tool, write_tool]
    )
    
    # Example 1: Code generation and execution
    print("Example 1: Generate and Execute Code\n")
    print("=" * 60)
    
    task1 = """Write Python code to calculate the fibonacci sequence up to n=10,
    then execute it and show the results."""
    
    print(f"Task: {task1}\n")
    
    result1 = agent.run(task1, max_steps=8)
    
    print(f"\nResult: {result1['result']}")
    print(f"Steps: {result1['total_steps']}")
    
    # Example 2: Code debugging
    print("\n\n" + "=" * 60)
    print("Example 2: Debug Code\n")
    print("=" * 60)
    
    buggy_code = """
def calculate_average(numbers):
    total = 0
    for num in numbers:
        total += num
    return total / len(numbers)

# This will cause an error
result = calculate_average([])
print(result)
"""
    
    task2 = f"""This code has a bug. Identify the issue and provide a fixed version:

{buggy_code}

Then execute the fixed version with test data."""
    
    print(f"Task: Debug and fix code\n")
    
    result2 = agent.run(task2, max_steps=10)
    
    print(f"\nResult: {result2['result']}")
    
    # Example 3: Code review with self-critique
    print("\n\n" + "=" * 60)
    print("Example 3: Code Review with Self-Critique\n")
    print("=" * 60)
    
    critique_agent = SelfCritiqueAgent(
        name="CodeReviewer",
        llm_client=llm_client
    )
    
    code_to_review = """
def process_data(data):
    result = []
    for item in data:
        if item > 0:
            result.append(item * 2)
    return result
"""
    
    reasoning_chain = [
        "The function takes a list as input",
        "It iterates through each item",
        "Positive numbers are doubled and added to result",
        "The function returns the processed list"
    ]
    
    critique = critique_agent.critique_reasoning(reasoning_chain)
    
    print(f"Code under review:")
    print(code_to_review)
    print(f"\nCritique:")
    print(critique['critique'])
    print(f"\nHas issues: {critique['has_issues']}")
    
    # Example 4: Create and save a utility function
    print("\n\n" + "=" * 60)
    print("Example 4: Create Utility Module\n")
    print("=" * 60)
    
    task4 = """Create a Python utility module with functions for:
    1. Converting celsius to fahrenheit
    2. Calculating compound interest
    3. Generating a random password

Save it to a file named 'utilities.py'"""
    
    print(f"Task: {task4}\n")
    
    result4 = agent.run(task4, max_steps=10)
    
    print(f"\nResult: {result4['result']}")
    print(f"Success: {result4['success']}")
    
    if result4['success']:
        print("\nUtility module created successfully!")
        print("Check ./output/utilities.py")


if __name__ == "__main__":
    main()