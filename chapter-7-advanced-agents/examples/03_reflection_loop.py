"""
Example 3: Reflection agent that self-evaluates and improves responses.
"""

import sys
sys.path.append('..')

from patterns.reflection_agent import ReflectionAgent, SelfCritiqueAgent
from anthropic import Anthropic
import os


def main():
    print("=== Reflection Agent Example ===\n")
    
    # Initialize LLM client
    llm_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    
    # Create reflection agent
    agent = ReflectionAgent(
        name="ReflectionAgent",
        llm_client=llm_client,
        max_reflections=3
    )
    
    # Task requiring high-quality output
    task = """Explain the concept of neural networks to a beginner. 
    Make it clear, accurate, and engaging."""
    
    print(f"Task: {task}\n")
    print("=" * 60 + "\n")
    
    # Run with reflection
    result = agent.run_with_reflection(task)
    
    print("=" * 60)
    print("\nReflection Process:\n")
    
    for i, reflection in enumerate(result['reflection_history'], 1):
        print(f"Iteration {i}:")
        print(f"Response excerpt: {reflection['response'][:150]}...")
        print(f"Reflection: {reflection['reflection'][:200]}...")
        print()
    
    print("=" * 60)
    print(f"\nFinal Response:\n{result['final_response']}")
    print(f"\nTotal Iterations: {result['iterations']}")
    
    # Example of self-critique
    print("\n" + "=" * 60)
    print("\nSelf-Critique Example:")
    
    critique_agent = SelfCritiqueAgent(
        name="CritiqueAgent",
        llm_client=llm_client
    )
    
    reasoning_chain = [
        "Neural networks are like the brain",
        "They have layers of neurons",
        "Each neuron processes information",
        "Therefore, they can think like humans"
    ]
    
    critique = critique_agent.critique_reasoning(reasoning_chain)
    
    print(f"\nReasoning Chain Critique:")
    print(f"Has Issues: {critique['has_issues']}")
    print(f"\nCritique:\n{critique['critique']}")


if __name__ == "__main__":
    main()