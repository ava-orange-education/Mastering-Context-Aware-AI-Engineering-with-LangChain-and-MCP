"""
Example 2: Knowledge Search

Demonstrates searching across enterprise knowledge base
"""

import asyncio
import sys
sys.path.append('../..')

from agents.document_search_agent import DocumentSearchAgent
from agents.summarization_agent import SummarizationAgent
from access_control.permission_manager import PermissionManager
from shared.utils import setup_logging

setup_logging()


async def basic_search():
    """Basic document search"""
    
    print("\n" + "="*70)
    print("Example 1: Basic Knowledge Search")
    print("="*70)
    
    # Initialize agent
    agent = DocumentSearchAgent()
    
    # Setup user permissions
    permission_manager = PermissionManager()
    user_id = "engineer@company.com"
    permission_manager.set_user_groups(user_id, ["engineering", "all_employees"])
    
    # Search
    query = "project planning strategy"
    print(f"\nQuery: {query}")
    print(f"User: {user_id}")
    print(f"Groups: {permission_manager.get_user_groups(user_id)}")
    
    print("\nSearching...")
    result = await agent.process({
        "query": query,
        "user_id": user_id,
        "user_groups": ["engineering", "all_employees"],
        "top_k": 5
    })
    
    print("\n" + "-"*70)
    print("Search Results")
    print("-"*70)
    
    print(f"\nConfidence: {result.confidence:.2%}")
    print(f"Sources found: {len(result.sources)}")
    
    if result.metadata.get("sources"):
        print("\nTop Documents:")
        for i, source in enumerate(result.metadata["sources"][:3], 1):
            print(f"\n{i}. {source.get('title', 'Untitled')}")
            print(f"   Source: {source.get('source', 'Unknown')}")
            print(f"   Relevance: {source.get('relevance_score', 0):.3f}")
    
    print("\n" + "-"*70)
    print("Answer")
    print("-"*70)
    print(result.content)


async def multi_source_search():
    """Search across multiple sources"""
    
    print("\n" + "="*70)
    print("Example 2: Multi-Source Search")
    print("="*70)
    
    agent = DocumentSearchAgent()
    
    user_id = "manager@company.com"
    query = "Q4 budget planning"
    
    print(f"\nQuery: {query}")
    print(f"User: {user_id}")
    print(f"Sources: SharePoint, Confluence, Slack")
    
    result = await agent.process({
        "query": query,
        "user_id": user_id,
        "user_groups": ["management", "finance"],
        "sources": ["sharepoint", "confluence", "slack"],
        "top_k": 10
    })
    
    print("\n" + "-"*70)
    print("Results by Source")
    print("-"*70)
    
    # Group by source
    from collections import defaultdict
    by_source = defaultdict(int)
    
    for source in result.sources:
        # Extract source from metadata
        by_source["confluence"] += 1  # Simplified
    
    for source, count in by_source.items():
        print(f"  {source}: {count} documents")
    
    print("\n" + "-"*70)
    print("Synthesized Answer")
    print("-"*70)
    print(result.content)


async def search_with_filters():
    """Search with metadata filters"""
    
    print("\n" + "="*70)
    print("Example 3: Filtered Search")
    print("="*70)
    
    agent = DocumentSearchAgent()
    
    user_id = "engineer@company.com"
    query = "API documentation"
    
    print(f"\nQuery: {query}")
    print(f"Filters:")
    print(f"  - Department: engineering")
    print(f"  - Document type: technical_documentation")
    
    result = await agent.process({
        "query": query,
        "user_id": user_id,
        "user_groups": ["engineering"],
        "filters": {
            "department": "engineering",
            "document_type": "technical_documentation"
        },
        "top_k": 5
    })
    
    print("\n" + "-"*70)
    print("Filtered Results")
    print("-"*70)
    print(result.content)


async def search_and_summarize():
    """Search and summarize results"""
    
    print("\n" + "="*70)
    print("Example 4: Search + Summarization")
    print("="*70)
    
    search_agent = DocumentSearchAgent()
    summarize_agent = SummarizationAgent()
    
    user_id = "product@company.com"
    query = "customer feedback Q1 2024"
    
    print(f"\nQuery: {query}")
    
    # Search
    print("\n1. Searching documents...")
    search_result = await search_agent.process({
        "query": query,
        "user_id": user_id,
        "user_groups": ["product", "customer_success"],
        "top_k": 5
    })
    
    print(f"   Found {len(search_result.sources)} relevant documents")
    
    # Summarize
    print("\n2. Summarizing results...")
    summary_result = await summarize_agent.process({
        "content": search_result.content,
        "document_type": "search_results",
        "max_length": 300
    })
    
    print("\n" + "-"*70)
    print("Summary")
    print("-"*70)
    print(summary_result.content)
    
    print("\n" + "-"*70)
    print("Key Information")
    print("-"*70)
    structured = summary_result.metadata.get("structured_info", {})
    print(f"  Has action items: {structured.get('has_action_items', False)}")
    print(f"  Has deadlines: {structured.get('has_deadlines', False)}")
    print(f"  Mentions stakeholders: {structured.get('mentions_stakeholders', False)}")


async def main():
    """Run all search examples"""
    
    print("\n" + "="*70)
    print("Enterprise Knowledge Search Examples")
    print("="*70)
    
    await basic_search()
    
    await multi_source_search()
    
    await search_with_filters()
    
    await search_and_summarize()
    
    print("\n" + "="*70)
    print("All Search Examples Complete!")
    print("="*70)


if __name__ == "__main__":
    asyncio.run(main())