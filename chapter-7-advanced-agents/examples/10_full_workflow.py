"""
Example 10: Complete end-to-end workflow with all features.
"""

import sys
sys.path.append('..')

from patterns.react_agent import ReActAgent
from patterns.planning_agent import PlanningAgent
from patterns.reflection_agent import ReflectionAgent
from multi_agent.agent_team import HierarchicalTeam, AgentRole
from memory.memory_manager import UnifiedMemoryManager
from tools.search_tools import WebSearchTool, VectorSearchTool
from tools.file_tools import WriteFileTool, ReadFileTool
from tools.tool_registry import ToolRegistry
from monitoring.agent_monitor import MultiAgentMonitor
from monitoring.trace_logger import TraceLogger
from safety.guardrails import SafetyGuardrails, SafetyRule, ContentSafetyFilter
from safety.cost_limiter import CostLimiter, RateLimiter
from mcp_integration.mcp_agent_server import MCPAgentServer
from embedding.embedding_manager import CachedEmbeddingManager
from retrieval.vector_store import ChromaVectorStore
from anthropic import Anthropic
import os
import time


def main():
    print("=" * 70)
    print("  COMPLETE AGENT WORKFLOW - ALL FEATURES DEMONSTRATION")
    print("=" * 70 + "\n")
    
    # ===== INITIALIZATION =====
    print("Phase 1: System Initialization")
    print("-" * 70)
    
    llm_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    
    embedding_mgr = CachedEmbeddingManager(
        model_name="text-embedding-3-small",
        provider="openai",
        cache_size=1000
    )
    
    vector_store = ChromaVectorStore(
        collection_name="full_workflow",
        dimensions=1536
    )
    
    print("✓ Core components initialized")
    
    # ===== MONITORING SETUP =====
    monitor = MultiAgentMonitor()
    trace_logger = TraceLogger(output_dir="./traces")
    
    print("✓ Monitoring systems ready")
    
    # ===== SAFETY SETUP =====
    guardrails = SafetyGuardrails()
    content_filter = ContentSafetyFilter()
    
    # Add safety rules
    guardrails.add_rule(SafetyRule(
        name="no_deletion",
        description="Prevent destructive operations",
        check_function=lambda data: "delete" not in str(data).lower(),
        severity="error"
    ))
    
    guardrails.add_rule(SafetyRule(
        name="content_safety",
        description="Check content safety",
        check_function=lambda data: content_filter.check_content(str(data))['safe'],
        severity="warning"
    ))
    
    cost_limiter = CostLimiter(daily_budget=5.0, monthly_budget=50.0)
    rate_limiter = RateLimiter(max_requests_per_minute=30)
    
    print("✓ Safety systems configured")
    
    # ===== TOOL SETUP =====
    tool_registry = ToolRegistry()
    
    search_tool = WebSearchTool()
    vector_search = VectorSearchTool(vector_store, embedding_mgr)
    write_tool = WriteFileTool(allowed_paths=["./output"])
    read_tool = ReadFileTool(allowed_paths=["./data"])
    
    tool_registry.register(search_tool, "search")
    tool_registry.register(vector_search, "search")
    tool_registry.register(write_tool, "file")
    tool_registry.register(read_tool, "file")
    
    print(f"✓ {len(tool_registry.tools)} tools registered")
    
    # ===== AGENT CREATION =====
    print("\nPhase 2: Agent Team Assembly")
    print("-" * 70)
    
    # Create specialized agents
    researcher = ReActAgent(
        name="ResearchAgent",
        llm_client=llm_client,
        tools=[search_tool, vector_search]
    )
    
    planner = PlanningAgent(
        name="PlannerAgent",
        llm_client=llm_client,
        tools=tool_registry.list_tools()
    )
    
    writer = ReflectionAgent(
        name="WriterAgent",
        llm_client=llm_client,
        tools=[write_tool],
        max_reflections=2
    )
    
    # Register for monitoring
    monitor.register_agent("ResearchAgent")
    monitor.register_agent("PlannerAgent")
    monitor.register_agent("WriterAgent")
    
    print("✓ Specialized agents created")
    
    # Create hierarchical team
    team = HierarchicalTeam("ProductionTeam", planner)
    
    team.add_agent(researcher, AgentRole(
        name="Researcher",
        description="Information gathering and analysis",
        capabilities=["research", "search", "analysis"],
        priority=3
    ))
    
    team.add_agent(writer, AgentRole(
        name="Writer",
        description="Content creation and refinement",
        capabilities=["writing", "editing", "reflection"],
        priority=2
    ))
    
    print("✓ Hierarchical team assembled")
    
    # ===== MEMORY SETUP =====
    memory = UnifiedMemoryManager(
        agent_id="workflow_manager",
        llm_client=llm_client,
        embedding_manager=embedding_mgr,
        vector_store=vector_store
    )
    
    print("✓ Memory systems initialized")
    
    # ===== MCP SERVER =====
    mcp_server = MCPAgentServer(planner, "workflow_mcp_server")
    
    print("✓ MCP server configured")
    print(f"  Server info: {mcp_server.get_server_info()}")
    
    # ===== WORKFLOW EXECUTION =====
    print("\n" + "=" * 70)
    print("Phase 3: Workflow Execution")
    print("=" * 70 + "\n")
    
    task = """Research the latest developments in AI agent architectures,
    create a comprehensive summary, and save it to a markdown file.
    
    Include:
    1. Key architectural patterns
    2. Recent innovations
    3. Best practices
    4. Future trends"""
    
    print(f"Task: {task}\n")
    
    # Safety check
    print("Step 1: Safety Validation")
    print("-" * 70)
    
    safety_check = guardrails.check_action("workflow_task", {"task": task})
    
    if not safety_check['allowed']:
        print("❌ Safety check failed:")
        for violation in safety_check['violations']:
            print(f"  - {violation['message']}")
        return
    
    print("✓ Safety check passed")
    
    if safety_check.get('warnings'):
        print(f"⚠ {len(safety_check['warnings'])} warnings:")
        for warning in safety_check['warnings']:
            print(f"  - {warning['message']}")
    
    # Budget check
    print("\nStep 2: Budget Verification")
    print("-" * 70)
    
    estimated_cost = 1.50
    budget_status = cost_limiter.check_budget()
    
    print(f"Estimated cost: ${estimated_cost:.2f}")
    print(f"Daily remaining: ${budget_status['daily']['remaining']:.2f}")
    print(f"Monthly remaining: ${budget_status['monthly']['remaining']:.2f}")
    
    if not cost_limiter.can_afford(estimated_cost):
        print("❌ Insufficient budget")
        return
    
    print("✓ Budget sufficient")
    
    # Rate limit check
    rate_limiter.wait_if_needed()
    rate_limiter.record_request()
    
    # Start trace
    trace_id = trace_logger.start_trace("WorkflowManager", task)
    print(f"\n✓ Execution trace started: {trace_id}")
    
    # Execute workflow
    print("\nStep 3: Team Execution")
    print("-" * 70)
    
    start_time = time.time()
    
    try:
        result = team.execute_task_hierarchical(task)
        execution_time = time.time() - start_time
        
        # Record actual cost
        actual_cost = 1.20
        cost_limiter.record_cost("workflow_execution", actual_cost)
        
        print(f"\n✓ Workflow completed in {execution_time:.2f}s")
        print(f"✓ Actual cost: ${actual_cost:.2f}")
        
        # Store in memory
        memory.add_interaction(
            user_input=task,
            agent_response=str(result.get('final_result')),
            success=result.get('success', False),
            metadata={
                'workflow': 'full_demo',
                'execution_time': execution_time,
                'cost': actual_cost
            }
        )
        
        # Finalize trace
        trace_logger.finalize_trace(trace_id, result, result.get('success', False))
        
        # Log monitoring data
        for agent_name in ["ResearchAgent", "PlannerAgent", "WriterAgent"]:
            agent_monitor = monitor.get_monitor(agent_name)
            if agent_monitor:
                agent_monitor.log_execution(
                    task=task,
                    result=result,
                    execution_time=execution_time / 3
                )
        
    except Exception as e:
        execution_time = time.time() - start_time
        print(f"\n❌ Workflow failed: {e}")
        
        trace_logger.finalize_trace(trace_id, {"error": str(e)}, False)
        return
    
    # ===== RESULTS AND ANALYSIS =====
    print("\n" + "=" * 70)
    print("Phase 4: Results and Analysis")
    print("=" * 70 + "\n")
    
    print("Execution Results:")
    print("-" * 70)
    print(f"Success: {result.get('success', False)}")
    print(f"Subtasks: {len(result.get('subtask_results', []))}")
    print(f"\nFinal Output Preview:")
    final_output = str(result.get('final_result', ''))
    print(final_output[:400] + "..." if len(final_output) > 400 else final_output)
    
    # Team statistics
    print("\n\nTeam Statistics:")
    print("-" * 70)
    team_status = team.get_team_status()
    print(f"Team: {team_status['team_name']}")
    print(f"Agents: {team_status['num_agents']}")
    print(f"Capabilities: {team_status['total_capabilities']}")
    
    # Monitoring statistics
    print("\n\nMonitoring Summary:")
    print("-" * 70)
    aggregate = monitor.get_aggregate_metrics()
    print(f"Total tasks: {aggregate['total_tasks']}")
    print(f"Success rate: {aggregate['overall_success_rate']:.2%}")
    print(f"Total errors: {aggregate['total_errors']}")
    
    # Cost summary
    print("\n\nCost Analysis:")
    print("-" * 70)
    cost_summary = cost_limiter.get_cost_summary()
    print(f"Total spent: ${cost_limiter.total_costs:.2f}")
    print(f"Daily budget used: {cost_summary['budget_status']['daily']['percentage']:.1f}%")
    print(f"Monthly budget used: {cost_summary['budget_status']['monthly']['percentage']:.1f}%")
    
    if cost_summary['top_expenses']:
        print("\nTop expenses:")
        for expense in cost_summary['top_expenses']:
            print(f"  {expense['operation']}: ${expense['cost']:.2f}")
    
    # Safety summary
    print("\n\nSafety Report:")
    print("-" * 70)
    safety_summary = guardrails.get_violation_summary()
    print(f"Total violations: {safety_summary['total_violations']}")
    print(f"By severity: {safety_summary['by_severity']}")
    
    # Memory statistics
    print("\n\nMemory Statistics:")
    print("-" * 70)
    memory_stats = memory.get_memory_stats()
    print(f"Short-term messages: {memory_stats['short_term_messages']}")
    print(f"Episodic experiences: {memory_stats['episodic_summary']['total_experiences']}")
    
    # Cache performance
    print("\n\nCache Performance:")
    print("-" * 70)
    cache_stats = embedding_mgr.get_cache_stats()
    print(f"Hit rate: {cache_stats['hit_rate']:.2%}")
    print(f"Cache size: {cache_stats['cache_size']}")
    print(f"Total requests: {cache_stats['cache_hits'] + cache_stats['cache_misses']}")
    
    # Trace summary
    print("\n\nExecution Trace:")
    print("-" * 70)
    trace_summary = trace_logger.get_trace_summary(trace_id)
    if trace_summary:
        print(f"Trace ID: {trace_summary['trace_id']}")
        print(f"Steps: {trace_summary['num_steps']}")
        print(f"Tool calls: {trace_summary['num_tool_calls']}")
        print(f"LLM calls: {trace_summary['num_llm_calls']}")
        print(f"Duration: {trace_summary['duration']:.2f}s")
    
    print("\n" + "=" * 70)
    print("  WORKFLOW COMPLETE")
    print("=" * 70)
    print(f"\nTrace saved to: ./traces/")
    print(f"Output files in: ./output/")


if __name__ == "__main__":
    main()