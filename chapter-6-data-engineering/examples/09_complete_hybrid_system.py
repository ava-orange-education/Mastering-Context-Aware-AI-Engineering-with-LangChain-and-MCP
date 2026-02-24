"""
Example: Complete hybrid RAG system implementation.
"""

import sys
sys.path.append('..')

from ingestion.database_connector import DatabaseConnector
from ingestion.pipeline_base import DataSource
from preprocessing.text_cleaner import TextCleaner, DocumentNormalizer
from preprocessing.chunking_strategies import DocumentChunker
from embedding.embedding_manager import CachedEmbeddingManager
from retrieval.vector_store import ChromaVectorStore
from retrieval.hybrid_retriever import BM25Index
from knowledge_graph.graph_builder import GraphBuilder
from hybrid_system.hybrid_rag import HybridRAGSystem
from monitoring.pipeline_monitor import PipelineMonitor
from anthropic import Anthropic
import os


def main():
    print("=== Building Complete Hybrid RAG System ===\n")
    
    # 1. Initialize components
    print("1. Initializing components...")
    
    # Embedding manager
    embedding_manager = CachedEmbeddingManager(
        model_name="text-embedding-3-small",
        provider="openai"
    )
    
    # Vector store
    vector_store = ChromaVectorStore(
        collection_name="hybrid_rag_demo",
        dimensions=1536
    )
    
    # Text cleaner and chunker
    text_cleaner = TextCleaner()
    chunker = DocumentChunker(strategy='sentence', chunk_size=512, chunk_overlap=50)
    
    # Knowledge graph builder
    graph_builder = GraphBuilder(
        entity_method="spacy",
        relation_method="llm"
    )
    
    # LLM client
    llm_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    
    print("✓ Components initialized\n")
    
    # 2. Ingest and process documents
    print("2. Ingesting sample documents...")
    
    sample_documents = [
        {
            'id': 'doc_1',
            'content': """
            Artificial Intelligence (AI) has transformed many industries. Machine learning,
            a subset of AI, enables systems to learn from data. Deep learning, using neural
            networks with multiple layers, has achieved remarkable results in image recognition
            and natural language processing. OpenAI developed GPT models for language tasks,
            while Google created BERT for understanding context.
            """,
            'metadata': {'source': 'ai_overview', 'category': 'technology'}
        },
        {
            'id': 'doc_2',
            'content': """
            Climate change poses significant challenges. Global temperatures have risen by
            approximately 1.1°C since pre-industrial times. The Paris Agreement aims to limit
            warming to 1.5°C. Renewable energy sources like solar and wind are crucial for
            reducing carbon emissions. Electric vehicles are becoming more popular as an
            alternative to fossil fuel-powered cars.
            """,
            'metadata': {'source': 'climate_report', 'category': 'environment'}
        },
        {
            'id': 'doc_3',
            'content': """
            Quantum computing leverages quantum mechanics to process information. Unlike
            classical bits, quantum bits (qubits) can exist in superposition. IBM and Google
            are leading quantum computing research. Quantum computers could solve complex
            optimization problems that are intractable for classical computers. Applications
            include drug discovery, cryptography, and financial modeling.
            """,
            'metadata': {'source': 'quantum_intro', 'category': 'technology'}
        }
    ]
    
    print(f"✓ Loaded {len(sample_documents)} documents\n")
    
    # 3. Process and embed documents
    print("3. Processing and embedding documents...")
    
    all_chunks = []
    all_texts = []
    all_metadata = []
    
    for doc in sample_documents:
        # Clean text
        cleaned = text_cleaner.clean(doc['content'])
        
        # Chunk document
        chunks = chunker.chunk_document({
            'content': cleaned,
            'metadata': doc['metadata']
        })
        
        all_chunks.extend(chunks)
        
        for chunk in chunks:
            all_texts.append(chunk.text)
            all_metadata.append({
                **chunk.metadata,
                'doc_id': doc['id']
            })
    
    # Generate embeddings
    print(f"   Generating embeddings for {len(all_chunks)} chunks...")
    embedding_results = embedding_manager.embed_batch(all_texts)
    embeddings = [r.embedding for r in embedding_results]
    
    print(f"✓ Processed {len(all_chunks)} chunks\n")
    
    # 4. Build vector index
    print("4. Building vector index...")
    vector_store.add_vectors(
        vectors=embeddings,
        metadata=all_metadata,
        ids=[f"chunk_{i}" for i in range(len(all_chunks))]
    )
    print("✓ Vector index built\n")
    
    # 5. Build BM25 index
    print("5. Building BM25 index...")
    bm25_index = BM25Index()
    bm25_index.build_index(all_texts, all_metadata)
    print("✓ BM25 index built\n")
    
    # 6. Build knowledge graph
    print("6. Building knowledge graph...")
    graph = graph_builder.build_from_documents(sample_documents)
    print(f"✓ Knowledge graph built: {len(graph.entities)} entities, {len(graph.relations)} relations\n")
    
    # 7. Create hybrid RAG system
    print("7. Creating hybrid RAG system...")
    hybrid_rag = HybridRAGSystem(
        vector_store=vector_store,
        bm25_index=bm25_index,
        knowledge_graph=graph,
        embedding_manager=embedding_manager,
        llm_client=llm_client
    )
    
    # Set fusion weights (can be tuned)
    hybrid_rag.set_fusion_weights(vector=0.5, bm25=0.3, graph=0.2)
    print("✓ Hybrid RAG system ready\n")
    
    # 8. Test queries
    print("8. Testing hybrid RAG system...\n")
    print("="*60 + "\n")
    
    test_queries = [
        "What is machine learning?",
        "How much have global temperatures risen?",
        "What companies are working on quantum computing?"
    ]
    
    for query in test_queries:
        print(f"Query: {query}\n")
        
        # Retrieve with hybrid approach
        result = hybrid_rag.query(query, top_k=3, use_graph=True)
        
        print(f"Answer: {result['answer']}\n")
        print(f"Sources ({result['num_sources']}):")
        for source in result['sources']:
            print(f"  [{source['index']}] Score: {source['score']:.3f}")
            print(f"      Category: {source['metadata'].get('category', 'unknown')}")
        
        print("\n" + "="*60 + "\n")
    
    # 9. Show cache statistics
    print("9. Performance Statistics:\n")
    cache_stats = embedding_manager.get_cache_stats()
    print(f"Embedding Cache:")
    print(f"  Hit Rate: {cache_stats['hit_rate']:.2%}")
    print(f"  Cache Size: {cache_stats['cache_size']}")
    print(f"  Total Requests: {cache_stats['cache_hits'] + cache_stats['cache_misses']}")


if __name__ == "__main__":
    main()