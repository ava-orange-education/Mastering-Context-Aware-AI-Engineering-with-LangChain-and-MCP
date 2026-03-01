"""
Example 5: Multiple agents collaborating on a shared task.
"""

import sys
sys.path.append('..')

from patterns.react_agent import ReActAgent
from patterns.planning_agent import PlanningAgent
from multi_agent.agent_team import AgentTeam, HierarchicalTeam, AgentRole
from multi_agent.communication import MessageBus, CommunicationProtocol
from tools.search_tools import WebSearchTool
from tools.file_tools import WriteFileTool
from anthropic import Anthropic
import os


def main():
    print("=== Multi-Agent Collaboration Example ===\n")
    
    # Initialize LLM client
    llm_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    
    # Create specialized agents
    researcher = ReActAgent(
        name="Researcher",
        llm_client=llm_client,
        tools=[WebSearchTool()]
    )
    
    analyzer = ReActAgent(
        name="Analyzer",
        llm_client=llm_client,
        tools=[]
    )
    
    writer = ReActAgent(
        name="Writer",
        llm_client=llm_client,
        tools=[WriteFileTool(allowed_paths=["./output"])]
    )
    
    # Example 1: Collaborative team
    print("Example 1: Collaborative Team\n")
    print("=" * 60)
    
    collab_team = AgentTeam("CollaborativeTeam")
    
    collab_team.add_agent(researcher, AgentRole(
        name="Researcher",
        description="Gathers information",
        capabilities=["research", "search"],
        priority=3
    ))
    
    collab_team.add_agent(analyzer, AgentRole(
        name="Analyzer",
        description="Analyzes information",
        capabilities=["analysis", "reasoning"],
        priority=2
    ))
    
    collab_team.add_agent(writer, AgentRole(
        name="Writer",
        description="Creates content",
        capabilities=["writing", "documentation"],
        priority=1
    ))
    
    task = "Research AI safety and create a brief summary"
    
    print(f"Task: {task}\n")
    
    result = collab_team.execute_task_collaborative(task, max_rounds=2)
    
    print(f"\nCollaborative Result:")
    print(f"Rounds: {len(result['rounds'])}")
    print(f"Final Output: {result['final_result'][:300]}...")
    
    # Example 2: Hierarchical team
    print("\n\n" + "=" * 60)
    print("Example 2: Hierarchical Team with Manager\n")
    print("=" * 60)
    
    manager = PlanningAgent(
        name="Manager",
        llm_client=llm_client,
        tools=[]
    )
    
    hierarchical_team = HierarchicalTeam("HierarchicalTeam", manager)
    
    hierarchical_team.add_agent(researcher, AgentRole(
        name="Researcher",
        description="Information gathering specialist",
        capabilities=["research"],
        priority=2
    ))
    
    hierarchical_team.add_agent(writer, AgentRole(
        name="Writer",
        description="Content creation specialist",
        capabilities=["writing"],
        priority=1
    ))
    
    task2 = "Create a technical report on machine learning deployment"
    
    print(f"Task: {task2}\n")
    
    result2 = hierarchical_team.execute_task_hierarchical(task2)
    
    print(f"\nHierarchical Result:")
    print(f"Subtasks executed: {len(result2['subtask_results'])}")
    print(f"Success: {result2['success']}")
    
    # Example 3: Agent communication
    print("\n\n" + "=" * 60)
    print("Example 3: Agent-to-Agent Communication\n")
    print("=" * 60)
    
    message_bus = MessageBus()
    protocol = CommunicationProtocol(message_bus)
    
    # Agent communication flow
    print("Setting up communication...")
    
    # Researcher requests information
    msg_id = protocol.request_information(
        requester="Researcher",
        provider="Analyzer",
        query="What are the key points to analyze?"
    )
    
    print(f"Researcher → Analyzer: Information request (ID: {msg_id})")
    
    # Analyzer responds
    protocol.respond_to_request(
        responder="Analyzer",
        original_message_id=msg_id,
        response_content="Focus on accuracy, relevance, and clarity"
    )
    
    print("Analyzer → Researcher: Response sent")
    
    # Check messages
    researcher_messages = message_bus.get_messages("Researcher")
    analyzer_messages = message_bus.get_messages("Analyzer")
    
    print(f"\nResearcher mailbox: {len(researcher_messages)} messages")
    print(f"Analyzer mailbox: {len(analyzer_messages)} messages")
    
    # Broadcast notification
    protocol.notify_agents(
        sender="Manager",
        recipients=["Researcher", "Analyzer", "Writer"],
        notification="Task completed successfully!"
    )
    
    print("\nManager → All: Broadcast notification sent")
    
    # Team status
    print("\n" + "=" * 60)
    print("\nTeam Status:")
    status = collab_team.get_team_status()
    print(f"Team: {status['team_name']}")
    print(f"Agents: {status['num_agents']}")
    print(f"Total Capabilities: {status['total_capabilities']}")
    print(f"Tasks Completed: {status['tasks_completed']}")


if __name__ == "__main__":
    main()