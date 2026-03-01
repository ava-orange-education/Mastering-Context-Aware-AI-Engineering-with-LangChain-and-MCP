"""
Example 6: Research agent with memory and learning capabilities.
"""

import sys
sys.path.append('..')

from patterns.react_agent import ReActAgent
from memory.memory_manager import UnifiedMemoryManager
from tools.search_tools import WebSearchTool, VectorSearchTool
from embedding.embedding_manager import CachedEmbeddingManager
from retrieval.vector_store import ChromaVectorStore
from anthropic import Anthropic
import os


def main():
    print("=== Research Agent with Memory ===\n")
    
    # Initialize components
    llm_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    
    embedding_mgr = CachedEmbeddingManager(
        model_name="text-embedding-3-small",
        provider="openai",
        cache_size=500
    )
    
    vector_store = ChromaVectorStore(
        collection_name="research_memory",
        dimensions=1536
    )
    
    # Initialize memory
    memory = UnifiedMemoryManager(
        agent_id="research_agent_001",
        llm_client=llm_client,
        embedding_manager=embedding_mgr,
        vector_store=vector_store
    )
    
    # Create tools
    web_search = WebSearchTool()
    vector_search = VectorSearchTool(vector_store, embedding_mgr)
    
    # Create agent
    agent = ReActAgent(
        name="ResearchAgent",
        llm_client=llm_client,
        tools=[web_search, vector_search]
    )
    
    # Research topics
    topics = [
        "What are transformers in machine learning?",
        "Explain attention mechanisms in neural networks",
        "How do large language models work?"
    ]
    
    print("Conducting Research on Multiple Topics\n")
    print("=" * 60 + "\n")
    
    # Research each topic
    for i, topic in enumerate(topics, 1):
        print(f"Research {i}/{len(topics)}: {topic}")
        print("-" * 60)
        
        # Get relevant context from memory
        context = memory.get_conversation_context(
            include_relevant_experiences=True,
            num_experiences=2
        )
        
        if context:
            print(f"Using context from memory: {len(context)} chars")
        
        # Conduct research
        result = agent.run(topic, max_steps=5)
        
        print(f"Result: {result['result'][:200]}...")
        
        # Store in memory
        memory.add_interaction(
            user_input=topic,
            agent_response=str(result['result']),
            success=result['success'],
            metadata={'research_topic': topic}
        )
        
        # Extract learnings
        memory.learn_from_interaction(
            f"Topic: {topic}\nFindings: {result['result']}"
        )
        
        print(f"Stored in memory\n")
    
    # Show memory statistics
    print("=" * 60)
    print("\nMemory Statistics:")
    
    stats = memory.get_memory_stats()
    print(f"Short-term messages: {stats['short_term_messages']}")
    print(f"Episodic experiences: {stats['episodic_summary']['total_experiences']}")
    print(f"Success rate: {stats['episodic_summary']['success_rate']:.2%}")
    
    # Demonstrate memory recall
    print("\n" + "=" * 60)
    print("\nMemory Recall Test:")
    
    query = "What did I learn about transformers?"
    print(f"\nQuery: {query}")
    
    # Recall from episodic memory
    similar_experiences = memory.episodic.recall_similar_experiences(
        query=query,
        top_k=3
    )
    
    print(f"\nFound {len(similar_experiences)} relevant experiences:")
    for i, exp in enumerate(similar_experiences, 1):
        exp_text = exp.get('metadata', {}).get('text', '')
        print(f"{i}. {exp_text[:150]}...")
    
    # Recall from semantic memory
    relevant_knowledge = memory.recall_relevant_knowledge(query, top_k=2)
    
    print(f"\nRelevant knowledge:")
    for i, knowledge in enumerate(relevant_knowledge, 1):
        print(f"{i}. {knowledge[:150]}...")
    
    # Show cache performance
    print("\n" + "=" * 60)
    print("\nEmbedding Cache Performance:")
    cache_stats = embedding_mgr.get_cache_stats()
    print(f"Hit rate: {cache_stats['hit_rate']:.2%}")
    print(f"Total requests: {cache_stats['cache_hits'] + cache_stats['cache_misses']}")
    print(f"Cache size: {cache_stats['cache_size']}")


if __name__ == "__main__":
    main()