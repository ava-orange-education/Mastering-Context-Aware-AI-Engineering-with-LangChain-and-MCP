"""
Example 9: Complete agent system with all features.
"""

import sys
sys.path.append('..')

from patterns.react_agent import ReActAgent
from patterns.planning_agent import PlanningAgent
from patterns.reflection_agent import ReflectionAgent
from multi_agent.agent_team import AgentTeam, AgentRole
from memory.memory_manager import UnifiedMemoryManager
from tools.search_tools import WebSearchTool, VectorSearchTool
from tools.file_tools import ReadFileTool, WriteFileTool
from tools.tool_registry import ToolRegistry
from monitoring.agent_monitor import MultiAgentMonitor
from safety.guardrails import SafetyGuardrails, SafetyRule
from safety.cost_limiter import CostLimiter
from embedding.embedding_manager import CachedEmbeddingManager
from retrieval.vector_store import ChromaVectorStore
from anthropic import Anthropic
import os


def main():
    print("=== Complete Agent System Example ===\n")
    
    # Initialize core components
    llm_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    
    embedding_mgr = CachedEmbeddingManager(
        model_name="text-embedding-3-small",
        provider="openai",
        cache_size=1000
    )
    
    vector_store = ChromaVectorStore(
        collection_name="complete_system",
        dimensions=1536
    )
    
    # Initialize monitoring
    monitor = MultiAgentMonitor()
    
    # Initialize safety
    guardrails = SafetyGuardrails()
    guardrails.add_rule(SafetyRule(
        name="no_file_deletion",
        description="Prevent file deletion",
        check_function=lambda data: "delete" not in str(data).lower(),
        severity="error",
        violation_message="File deletion is not allowed"
    ))
    
    cost_limiter = CostLimiter(daily_budget=10.0, monthly_budget=100.0)
    
    # Initialize tool registry
    tool_registry = ToolRegistry()
    
    # Register tools
    search_tool = WebSearchTool()
    vector_search = VectorSearchTool(vector_store, embedding_mgr)
    read_tool = ReadFileTool(allowed_paths=["./data"])
    write_tool = WriteFileTool(allowed_paths=["./output"])
    
    tool_registry.register(search_tool, "search")
    tool_registry.register(vector_search, "search")
    tool_registry.register(read_tool, "file")
    tool_registry.register(write_tool, "file")
    
    # Create agents with different specializations
    research_agent = ReActAgent(
        name="ResearchAgent",
        llm_client=llm_client,
        tools=[search_tool, vector_search]
    )
    
    planning_agent = PlanningAgent(
        name="PlanningAgent",
        llm_client=llm_client,
        tools=tool_registry.list_tools()
    )
    
    writing_agent = ReflectionAgent(
        name="WritingAgent",
        llm_client=llm_client,
        tools=[write_tool],
        max_reflections=2
    )
    
    # Register agents for monitoring
    research_monitor = monitor.register_agent("ResearchAgent")
    planning_monitor = monitor.register_agent("PlanningAgent")
    writing_monitor = monitor.register_agent("WritingAgent")
    
    # Create agent team
    team = AgentTeam("ContentCreationTeam")
    
    team.add_agent(research_agent, AgentRole(
        name="Researcher",
        description="Gathers and analyzes information",
        capabilities=["research", "analysis"],
        priority=2
    ))
    
    team.add_agent(planning_agent, AgentRole(
        name="Planner",
        description="Creates and executes plans",
        capabilities=["planning", "coordination"],
        priority=3
    ))
    
    team.add_agent(writing_agent, AgentRole(
        name="Writer",
        description="Creates polished content",
        capabilities=["writing", "editing"],
        priority=1
    ))
    
    # Execute collaborative task
    task = """Create a brief article about recent advances in AI agents.
    
Steps:
1. Research recent developments
2. Plan article structure
3. Write and refine content"""
    
    print(f"Task: {task}\n")
    print("=" * 60 + "\n")
    
    # Check safety
    safety_check = guardrails.check_action("collaborative_task", {"task": task})
    
    if not safety_check['allowed']:
        print("Safety check failed:")
        for violation in safety_check['violations']:
            print(f"  - {violation['message']}")
        return
    
    # Check budget
    estimated_cost = 0.50  # Estimate
    
    if not cost_limiter.can_afford(estimated_cost):
        print("Insufficient budget for this task")
        print(cost_limiter.get_cost_summary())
        return
    
    # Execute task collaboratively
    import time
    start_time = time.time()
    
    result = team.execute_task_collaborative(task, max_rounds=2)
    
    execution_time = time.time() - start_time
    
    # Record costs
    actual_cost = 0.35  # Actual cost after execution
    cost_limiter.record_cost("collaborative_task", actual_cost)
    
    # Log execution for each agent
    for agent_name in ["ResearchAgent", "PlanningAgent", "WritingAgent"]:
        agent_monitor = monitor.get_monitor(agent_name)
        if agent_monitor:
            agent_monitor.log_execution(
                task=task,
                result=result,
                execution_time=execution_time / 3  # Distribute time
            )
    
    print("\n" + "=" * 60)
    print("\nTask Complete!\n")
    
    # Print results
    print("Final Result:")
    print(result['final_result'][:500] + "..." if len(result['final_result']) > 500 else result['final_result'])
    
    print("\n" + "=" * 60)
    print("\nSystem Statistics:\n")
    
    # Team status
    team_status = team.get_team_status()
    print(f"Team: {team_status['team_name']}")
    print(f"Agents: {team_status['num_agents']}")
    print(f"Tasks completed: {team_status['tasks_completed']}")
    
    # Monitoring stats
    aggregate_metrics = monitor.get_aggregate_metrics()
    print(f"\nOverall Success Rate: {aggregate_metrics['overall_success_rate']:.2%}")
    print(f"Total Tasks: {aggregate_metrics['total_tasks']}")
    
    # Cost summary
    cost_summary = cost_limiter.get_cost_summary()
    print(f"\nCost Summary:")
    print(f"  Daily spent: ${cost_summary['budget_status']['daily']['spent']:.2f}")
    print(f"  Daily remaining: ${cost_summary['budget_status']['daily']['remaining']:.2f}")
    
    # Safety summary
    safety_summary = guardrails.get_violation_summary()
    print(f"\nSafety:")
    print(f"  Total violations: {safety_summary['total_violations']}")
    
    # Cache stats
    cache_stats = embedding_mgr.get_cache_stats()
    print(f"\nEmbedding Cache:")
    print(f"  Hit rate: {cache_stats['hit_rate']:.2%}")
    print(f"  Cache size: {cache_stats['cache_size']}")


if __name__ == "__main__":
    main()