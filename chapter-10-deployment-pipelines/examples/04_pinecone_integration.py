"""
Example 4: Pinecone Integration
Demonstrates Pinecone vector store usage
"""

import asyncio
import os
from vector_stores.pinecone_store import PineconeStore
from vector_stores.base_store import Document


async def initialize_pinecone():
    """Initialize Pinecone connection"""
    print("\n" + "="*70)
    print("Initializing Pinecone")
    print("="*70)
    
    # Check environment variables
    if not os.getenv("PINECONE_API_KEY"):
        print("❌ PINECONE_API_KEY not set in environment")
        print("Set it with: export PINECONE_API_KEY=your-key")
        return None
    
    store = PineconeStore()
    await store.initialize()
    
    print("✅ Pinecone initialized")
    return store


async def create_index(store):
    """Create a new index"""
    print("\n" + "="*70)
    print("Creating Index")
    print("="*70)
    
    dimension = 1536  # Standard embedding dimension
    
    await store.create_index(
        dimension=dimension,
        metric="cosine"
    )
    
    print(f"✅ Index created (dimension={dimension})")


async def upsert_documents(store):
    """Insert sample documents"""
    print("\n" + "="*70)
    print("Upserting Documents")
    print("="*70)
    
    # Create sample documents with embeddings
    documents = []
    
    texts = [
        "Machine learning is a subset of artificial intelligence.",
        "Deep learning uses neural networks with multiple layers.",
        "Natural language processing enables computers to understand human language.",
        "Computer vision allows machines to interpret visual information.",
        "Reinforcement learning trains agents through rewards and penalties."
    ]
    
    for i, text in enumerate(texts):
        # Generate simple embedding (in production, use real embedding model)
        embedding = [float(ord(c)) / 255.0 for c in text[:1536]]
        # Pad to 1536 dimensions
        embedding.extend([0.0] * (1536 - len(embedding)))
        
        doc = Document(
            id=f"doc_{i}",
            content=text,
            embedding=embedding,
            metadata={"source": "example", "index": i}
        )
        documents.append(doc)
    
    await store.upsert(documents)
    
    print(f"✅ Upserted {len(documents)} documents")


async def search_documents(store):
    """Search for similar documents"""
    print("\n" + "="*70)
    print("Searching Documents")
    print("="*70)
    
    query = "What is neural network?"
    
    # Generate query embedding
    query_embedding = [float(ord(c)) / 255.0 for c in query[:1536]]
    query_embedding.extend([0.0] * (1536 - len(query_embedding)))
    
    results = await store.search(
        query_embedding=query_embedding,
        top_k=3
    )
    
    print(f"\nQuery: {query}")
    print(f"Found {len(results)} results:\n")
    
    for i, result in enumerate(results, 1):
        print(f"{i}. Score: {result.score:.4f}")
        print(f"   Content: {result.document.content}")
        print(f"   Metadata: {result.document.metadata}")
        print()


async def get_stats(store):
    """Get index statistics"""
    print("\n" + "="*70)
    print("Index Statistics")
    print("="*70)
    
    stats = await store.get_stats()
    
    print(f"\nTotal vectors: {stats.get('total_vector_count', 0)}")
    print(f"Dimension: {stats.get('dimension', 0)}")
    print(f"Index fullness: {stats.get('index_fullness', 0):.2%}")


async def delete_documents(store):
    """Delete documents"""
    print("\n" + "="*70)
    print("Deleting Documents")
    print("="*70)
    
    doc_ids = ["doc_0", "doc_1"]
    
    await store.delete(doc_ids)
    
    print(f"✅ Deleted {len(doc_ids)} documents")


async def health_check(store):
    """Check health"""
    print("\n" + "="*70)
    print("Health Check")
    print("="*70)
    
    is_healthy = await store.health_check()
    
    if is_healthy:
        print("✅ Pinecone is healthy")
    else:
        print("❌ Pinecone health check failed")


async def main():
    """Main function"""
    print("\n" + "="*70)
    print("Pinecone Integration Examples")
    print("="*70)
    
    # Initialize
    store = await initialize_pinecone()
    
    if not store:
        return
    
    try:
        # Create index (if needed)
        # Uncomment if you need to create a new index
        # await create_index(store)
        
        # Health check
        await health_check(store)
        
        # Upsert documents
        await upsert_documents(store)
        
        # Get stats
        await get_stats(store)
        
        # Search
        await search_documents(store)
        
        # Delete some documents
        # await delete_documents(store)
        
        print("\n" + "="*70)
        print("All operations completed successfully!")
        print("="*70)
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
    
    finally:
        await store.close()


if __name__ == "__main__":
    asyncio.run(main())