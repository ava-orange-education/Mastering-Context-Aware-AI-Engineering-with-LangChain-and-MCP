"""
Example 2: Langfuse Tracing
Demonstrates Langfuse integration for observability.
"""

import sys
sys.path.append('..')

from observability.langfuse_integration import LangfuseIntegration
import os


def main():
    print("=" * 70)
    print("  LANGFUSE TRACING EXAMPLE")
    print("=" * 70 + "\n")
    
    # Initialize Langfuse (requires API keys)
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY")
    
    langfuse = LangfuseIntegration(
        public_key=public_key,
        secret_key=secret_key
    )
    
    if not langfuse.enabled:
        print("⚠️  Langfuse not configured. Set LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY")
        print("    This example will demonstrate the API without actual tracing.\n")
    
    # Example 1: Trace agent run
    print("1. TRACING AGENT RUN")
    print("-" * 70)
    
    query = "What is machine learning?"
    response = "Machine learning is a subset of AI that enables systems to learn from data."
    
    trace_id = langfuse.trace_agent_run(
        agent_name="QuestionAnswering",
        query=query,
        response=response,
        metadata={'model': 'claude-3-5-sonnet', 'temperature': 0.7}
    )
    
    if trace_id:
        print(f"✓ Agent run traced: {trace_id}")
    else:
        print("  Agent run would be traced with ID: trace_12345")
    
    print(f"  Query: {query}")
    print(f"  Response: {response[:60]}...")
    
    # Example 2: Trace LLM generation
    print("\n2. TRACING LLM GENERATION")
    print("-" * 70)
    
    generation_id = langfuse.trace_generation(
        model="claude-3-5-sonnet-20241022",
        prompt="Explain quantum computing",
        completion="Quantum computing uses quantum mechanics principles...",
        token_count=150,
        latency_ms=1250
    )
    
    if generation_id:
        print(f"✓ Generation traced: {generation_id}")
    else:
        print("  Generation would be traced with ID: gen_67890")
    
    print(f"  Model: claude-3-5-sonnet-20241022")
    print(f"  Tokens: 150")
    print(f"  Latency: 1250ms")
    
    # Example 3: Trace retrieval
    print("\n3. TRACING RETRIEVAL")
    print("-" * 70)
    
    documents = [
        {'id': 'doc1', 'content': 'Machine learning basics...', 'score': 0.95},
        {'id': 'doc2', 'content': 'Deep learning overview...', 'score': 0.87},
        {'id': 'doc3', 'content': 'Neural networks intro...', 'score': 0.82}
    ]
    
    retrieval_id = langfuse.trace_retrieval(
        query="machine learning basics",
        documents=documents,
        metadata={'vector_db': 'chromadb', 'top_k': 3}
    )
    
    if retrieval_id:
        print(f"✓ Retrieval traced: {retrieval_id}")
    else:
        print("  Retrieval would be traced with ID: ret_11111")
    
    print(f"  Retrieved: {len(documents)} documents")
    print(f"  Top score: {documents[0]['score']}")
    
    # Example 4: Log evaluation scores
    print("\n4. LOGGING EVALUATION SCORES")
    print("-" * 70)
    
    if trace_id:
        langfuse.log_score(
            trace_id=trace_id,
            name="relevance",
            value=0.92,
            comment="High relevance to query"
        )
        
        langfuse.log_score(
            trace_id=trace_id,
            name="coherence",
            value=0.88,
            comment="Well-structured response"
        )
        
        print(f"✓ Scores logged for trace {trace_id}")
    else:
        print("  Scores would be logged:")
        print("    - relevance: 0.92")
        print("    - coherence: 0.88")
    
    # Flush traces
    print("\n5. FLUSHING TRACES")
    print("-" * 70)
    
    langfuse.flush()
    print("✓ All pending traces flushed to Langfuse")
    
    print("\n" + "=" * 70)
    print("  LANGFUSE TRACING DEMONSTRATION COMPLETE")
    print("=" * 70)
    
    if not langfuse.enabled:
        print("\n💡 To enable actual tracing:")
        print("   1. Sign up at https://langfuse.com")
        print("   2. Get your API keys")
        print("   3. Set environment variables:")
        print("      export LANGFUSE_PUBLIC_KEY='your-public-key'")
        print("      export LANGFUSE_SECRET_KEY='your-secret-key'")


if __name__ == "__main__":
    main()