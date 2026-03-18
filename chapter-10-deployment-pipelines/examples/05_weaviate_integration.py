"""
Example 5: Weaviate Integration
Demonstrates Weaviate vector store usage
"""

import asyncio
import os
from vector_stores.weaviate_store import WeaviateStore
from vector_stores.base_store import Document


async def initialize_weaviate():
    """Initialize Weaviate connection"""
    print("\n" + "="*70)
    print("Initializing Weaviate")
    print("="*70)
    
    # Check environment variables
    if not os.getenv("WEAVIATE_URL"):
        print("❌ WEAVIATE_URL not set in environment")
        print("Set it with: export WEAVIATE_URL=your-url")
        return None
    
    store = WeaviateStore()
    await store.initialize()
    
    print("✅ Weaviate initialized")
    return store


async def create_schema(store):
    """Create collection schema"""
    print("\n" + "="*70)
    print("Creating Schema")
    print("="*70)
    
    dimension = 1536
    
    await store.create_index(
        dimension=dimension,
        vectorizer="none"
    )
    
    print(f"✅ Schema created (dimension={dimension})")


async def upsert_documents(store):
    """Insert sample documents"""
    print("\n" + "="*70)
    print("Upserting Documents")
    print("="*70)
    
    documents = []
    
    texts = [
        "Weaviate is a cloud-native vector database.",
        "Vector databases enable semantic search capabilities.",
        "Embeddings represent data as numerical vectors.",
        "Similarity search finds semantically related content.",
        "Hybrid search combines vector and keyword search."
    ]
    
    for i, text in enumerate(texts):
        # Generate simple embedding
        embedding = [float(ord(c)) / 255.0 for c in text[:1536]]
        embedding.extend([0.0] * (1536 - len(embedding)))
        
        doc = Document(
            id=f"weaviate_doc_{i}",
            content=text,
            embedding=embedding,
            metadata={
                "source": "example",
                "category": "vector_db",
                "index": i
            }
        )
        documents.append(doc)
    
    await store.upsert(documents)
    
    print(f"✅ Upserted {len(documents)} documents")


async def search_documents(store):
    """Search for similar documents"""
    print("\n" + "="*70)
    print("Searching Documents")
    print("="*70)
    
    query = "What is semantic search?"
    
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


async def filtered_search(store):
    """Search with metadata filtering"""
    print("\n" + "="*70)
    print("Filtered Search")
    print("="*70)
    
    query_embedding = [0.5] * 1536
    
    # Search with filter
    filter_clause = {
        "path": ["metadata", "category"],
        "operator": "Equal",
        "valueString": "vector_db"
    }
    
    results = await store.search(
        query_embedding=query_embedding,
        top_k=5,
        filter=filter_clause
    )
    
    print(f"Found {len(results)} results with filter")
    
    for result in results:
        print(f"- {result.document.content}")


async def get_stats(store):
    """Get collection statistics"""
    print("\n" + "="*70)
    print("Collection Statistics")
    print("="*70)
    
    stats = await store.get_stats()
    
    print(f"\nTotal objects: {stats.get('total_objects', 0)}")
    print(f"Class name: {stats.get('class_name', 'N/A')}")


async def health_check(store):
    """Check health"""
    print("\n" + "="*70)
    print("Health Check")
    print("="*70)
    
    is_healthy = await store.health_check()
    
    if is_healthy:
        print("✅ Weaviate is healthy")
    else:
        print("❌ Weaviate health check failed")


async def main():
    """Main function"""
    print("\n" + "="*70)
    print("Weaviate Integration Examples")
    print("="*70)
    
    # Initialize
    store = await initialize_weaviate()
    
    if not store:
        return
    
    try:
        # Create schema (if needed)
        # await create_schema(store)
        
        # Health check
        await health_check(store)
        
        # Upsert documents
        await upsert_documents(store)
        
        # Get stats
        await get_stats(store)
        
        # Search
        await search_documents(store)
        
        # Filtered search
        await filtered_search(store)
        
        print("\n" + "="*70)
        print("All operations completed successfully!")
        print("="*70)
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        await store.close()


if __name__ == "__main__":
    asyncio.run(main())