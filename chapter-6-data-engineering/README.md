# Chapter 6: Data Engineering for Context-Aware AI

A comprehensive data engineering framework for building production-ready context-aware AI systems with multi-modal data ingestion, preprocessing, embedding generation, hybrid retrieval, and knowledge graph integration.

## 🌟 Features

### Data Ingestion
- **Multi-Source Connectors**: PostgreSQL, BigQuery, REST APIs, local files, S3
- **Incremental Processing**: Checkpoint-based extraction for efficient updates
- **Rate Limiting**: Built-in rate limiting for API sources
- **Flexible Pagination**: Support for offset, cursor, and page-based pagination

### Preprocessing & Chunking
- **Text Cleaning**: Unicode normalization, HTML/Markdown removal, pattern-based cleaning
- **Multiple Chunking Strategies**:
  - Fixed-size chunking with overlap
  - Sentence-based chunking
  - Semantic chunking using embeddings
  - Hierarchical chunking with parent-child relationships
- **Document Normalization**: Unified processing for PDF, DOCX, HTML, JSON

### Embedding Management
- **Multiple Providers**: OpenAI, Sentence Transformers, Cohere, Voyage AI
- **Caching Layer**: Reduces redundant API calls and costs
- **Batch Processing**: Efficient parallel embedding generation
- **Model Optimization**: Support for quantization and fine-tuning

### Vector & Hybrid Retrieval
- **Vector Stores**: ChromaDB, Pinecone, Weaviate integration
- **Hybrid Search**: Combines dense (vector) and sparse (BM25) retrieval
- **Result Fusion**: Reciprocal Rank Fusion for combining multiple sources
- **Query Optimization**: Intelligent routing and reranking

### Knowledge Graph
- **Entity Extraction**: spaCy NER, LLM-based, pattern-based extraction
- **Relation Extraction**: Dependency parsing and LLM-based relation discovery
- **Graph Querying**: Path finding, clustering, subgraph extraction, pattern matching
- **Neo4j Export**: Direct export to Neo4j graph database

### Pipeline Orchestration
- **Airflow Integration**: DAG generation for scheduled pipelines
- **Checkpoint Management**: Incremental processing with state persistence
- **Error Handling**: Exponential backoff and graceful degradation
- **Monitoring**: Real-time metrics, health checks, and alerting

### Production Features
- **Data Quality Monitoring**: Completeness, uniqueness, consistency, accuracy, timeliness checks
- **Performance Tracking**: Latency metrics, throughput analysis, error rate monitoring
- **Cost Optimization**: Budget tracking, batch processing, efficient caching
- **Comprehensive Logging**: Structured logging throughout the pipeline

## 📋 Prerequisites

- Python 3.8 or higher
- PostgreSQL (optional, for database connector)
- Neo4j (optional, for knowledge graph export)
- API keys for:
  - OpenAI (for embeddings and LLM)
  - Anthropic (for Claude integration)
  - Cohere (optional, for Cohere embeddings)
  - Voyage AI (optional, for Voyage embeddings)

## 🚀 Quick Start

### Installation

1. Clone the repository:
```bash
git clone https://github.com/ava-orange-education/Mastering-Context-Aware-AI-Engineering-with-LangChain-and-MCP.git
cd Mastering-Context-Aware-AI-Engineering-with-LangChain-and-MCP/chapter-6-data-engineering
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download spaCy model (for entity extraction):
```bash
python -m spacy download en_core_web_sm
```

5. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys and configuration
```

### Basic Usage

#### 1. Data Ingestion
```python
from ingestion.database_connector import DatabaseConnector
from ingestion.pipeline_base import DataSource

# Configure data source
source = DataSource(
    source_id="my_database",
    source_type="database",
    connection_params={
        'host': 'localhost',
        'database': 'mydb',
        'user': 'user',
        'password': 'password',
        'table': 'documents',
        'timestamp_column': 'updated_at'
    }
)

# Create connector
connector = DatabaseConnector(source)

# Extract data
if connector.connect():
    for record in connector.extract():
        print(f"Extracted: {record.record_id}")
    connector.close()
```

#### 2. Text Preprocessing & Chunking
```python
from preprocessing.text_cleaner import TextCleaner
from preprocessing.chunking_strategies import DocumentChunker

# Clean text
cleaner = TextCleaner(config={
    'lowercase': False,
    'remove_urls': True,
    'remove_extra_whitespace': True
})

cleaned_text = cleaner.clean(raw_text)

# Chunk document
chunker = DocumentChunker(
    strategy='sentence',  # or 'fixed', 'semantic', 'hierarchical'
    chunk_size=512,
    chunk_overlap=50
)

chunks = chunker.chunk_document({
    'content': cleaned_text,
    'metadata': {'source': 'my_doc'}
})
```

#### 3. Embedding Generation
```python
from embedding.embedding_manager import CachedEmbeddingManager

# Initialize embedding manager with caching
embedding_manager = CachedEmbeddingManager(
    model_name="text-embedding-3-small",
    provider="openai",
    cache_size=10000
)

# Generate embeddings
texts = [chunk.text for chunk in chunks]
results = embedding_manager.embed_batch(texts)

# Check cache statistics
stats = embedding_manager.get_cache_stats()
print(f"Cache hit rate: {stats['hit_rate']:.2%}")
```

#### 4. Vector Storage & Retrieval
```python
from retrieval.vector_store import ChromaVectorStore

# Initialize vector store
vector_store = ChromaVectorStore(
    collection_name="my_documents",
    dimensions=1536,
    persist_directory="./chroma_db"
)

# Add vectors
embeddings = [r.embedding for r in results]
metadata = [{'text': chunk.text, 'chunk_id': chunk.chunk_id} for chunk in chunks]
vector_store.add_vectors(embeddings, metadata)

# Search
query_embedding = embedding_manager.embed_text("What is machine learning?").embedding
results = vector_store.search(query_embedding, top_k=5)
```

#### 5. Hybrid Retrieval
```python
from retrieval.hybrid_retriever import HybridRetriever, BM25Index

# Build BM25 index
bm25_index = BM25Index()
documents = [chunk.text for chunk in chunks]
bm25_index.build_index(documents, metadata)

# Create hybrid retriever
hybrid_retriever = HybridRetriever(
    vector_store=vector_store,
    bm25_index=bm25_index,
    alpha=0.5  # Weight for dense vs sparse
)

# Hybrid search
results = hybrid_retriever.search(
    query="machine learning applications",
    query_vector=query_embedding,
    top_k=10
)
```

#### 6. Knowledge Graph Construction
```python
from knowledge_graph.graph_builder import GraphBuilder

# Build knowledge graph
graph_builder = GraphBuilder(
    entity_method="spacy",
    relation_method="llm"
)

# Process documents
documents = [
    {'id': 'doc_1', 'content': 'OpenAI developed GPT-4...'},
    {'id': 'doc_2', 'content': 'Google created BERT...'}
]

graph = graph_builder.build_from_documents(documents)

print(f"Entities: {len(graph.entities)}")
print(f"Relations: {len(graph.relations)}")

# Query graph
from knowledge_graph.graph_querying import GraphQuery

graph_query = GraphQuery(graph)
path = graph_query.find_path(source_id, target_id)
related = graph_query.find_related_entities(entity_id, max_depth=2)
```

#### 7. Complete Hybrid RAG System
```python
from hybrid_system.hybrid_rag import HybridRAGSystem
from anthropic import Anthropic

# Initialize hybrid RAG
hybrid_rag = HybridRAGSystem(
    vector_store=vector_store,
    bm25_index=bm25_index,
    knowledge_graph=graph,
    embedding_manager=embedding_manager,
    llm_client=Anthropic(api_key="your-key")
)

# Set fusion weights
hybrid_rag.set_fusion_weights(vector=0.5, bm25=0.3, graph=0.2)

# Query
result = hybrid_rag.query(
    question="What is quantum computing?",
    top_k=5,
    use_graph=True
)

print(f"Answer: {result['answer']}")
print(f"Sources: {result['num_sources']}")
```

#### 8. Pipeline Orchestration
```python
from pipeline.orchestrator import PipelineOrchestrator

# Configure pipeline
config = {
    'pipeline_id': 'my_data_pipeline',
    'schedule': '@daily'
}

orchestrator = PipelineOrchestrator(config)

# Register components
orchestrator.register_ingestion_connector(connector)
orchestrator.set_embedding_manager(embedding_manager)
orchestrator.set_vector_store(vector_store)
orchestrator.set_graph_builder(graph_builder)

# Run pipeline
checkpoint_store = {}
stats = orchestrator.run_pipeline(checkpoint_store)

print(f"Pipeline Status: {stats['status']}")
print(f"Records Processed: {stats['records_processed']}")
print(f"Embeddings Generated: {stats['embeddings_generated']}")
```

#### 9. Monitoring & Quality Checks
```python
from monitoring.quality_monitor import QualityMonitor, CompletenessRule, UniquenessRule
from monitoring.pipeline_monitor import PipelineMonitor

# Set up quality monitoring
quality_monitor = QualityMonitor()
quality_monitor.add_rule(CompletenessRule(required_fields=['content', 'metadata']))
quality_monitor.add_rule(UniquenessRule(key_fields=['record_id']))

# Validate records
validation_result = quality_monitor.validate_batch(records)
print(f"Pass Rate: {validation_result['pass_rate']:.2%}")

# Set up pipeline monitoring
pipeline_monitor = PipelineMonitor()
pipeline_monitor.set_alert_threshold('error_rate', 0.05)

# Monitor execution
result = pipeline_monitor.monitor_execution(
    pipeline_func=orchestrator.run_pipeline,
    pipeline_id='my_pipeline',
    checkpoint_store=checkpoint_store
)

# Check health
health = pipeline_monitor.get_health_status()
print(f"Pipeline Status: {health['status']}")
print(f"Success Rate: {health['success_rate']:.2%}")
```

## 📚 Examples

The `examples/` directory contains 9 comprehensive examples:

1. **01_basic_ingestion.py** - Basic data ingestion from multiple sources
2. **02_text_preprocessing.py** - Text cleaning and normalization
3. **03_embedding_generation.py** - Embedding generation with caching
4. **04_vector_search.py** - Vector similarity search
5. **05_knowledge_graph.py** - Knowledge graph construction and querying
6. **06_hybrid_retrieval.py** - Hybrid search combining vector and keyword
7. **07_pipeline_orchestration.py** - Complete pipeline orchestration
8. **08_monitoring_setup.py** - Quality monitoring and alerting
9. **09_complete_hybrid_system.py** - End-to-end hybrid RAG system

Run any example:
```bash
python examples/01_basic_ingestion.py
```

## 🏗️ Architecture
```
┌─────────────────────────────────────────────────┐
│           Pipeline Orchestrator                  │
│         (Airflow DAG Generation)                │
└─────────┬───────────────────────────────────────┘
          │
    ┌─────▼─────┐
    │ Ingestion │
    │ Connectors│
    └─────┬─────┘
          │
    ┌─────▼──────┐
    │Preprocessor│
    │  & Chunker │
    └─────┬──────┘
          │
    ┌─────▼─────────┐
    │   Embedding   │
    │   Generation  │
    └─────┬─────────┘
          │
    ┌─────▼─────────┬──────────────┐
    │               │              │
┌───▼────┐  ┌──────▼─────┐  ┌────▼─────┐
│ Vector │  │    BM25    │  │Knowledge │
│ Store  │  │   Index    │  │  Graph   │
└───┬────┘  └──────┬─────┘  └────┬─────┘
    │              │              │
    └──────┬───────┴──────────────┘
           │
    ┌──────▼──────┐
    │   Hybrid    │
    │ RAG System  │
    └─────────────┘
```

## 🔧 Configuration

Configuration files are located in `configs/`:

### `pipeline_config.yaml`
```yaml
pipeline:
  pipeline_id: "production_data_pipeline"
  schedule: "@daily"
  
ingestion:
  batch_size: 1000
  max_retries: 3
  
preprocessing:
  chunking_strategy: "sentence"
  chunk_size: 512
  chunk_overlap: 50
  
embedding:
  provider: "openai"
  model: "text-embedding-3-small"
  batch_size: 100
  
vector_store:
  provider: "chroma"
  collection_name: "documents"
  persist_directory: "./chroma_db"
```

### `embedding_config.yaml`
```yaml
providers:
  openai:
    model: "text-embedding-3-small"
    dimensions: 1536
    
  sentence_transformers:
    model: "all-MiniLM-L6-v2"
    dimensions: 384
    
  cohere:
    model: "embed-english-v3.0"
    dimensions: 1024

caching:
  enabled: true
  cache_size: 10000
  ttl_seconds: 86400
```

### `monitoring_config.yaml`
```yaml
quality_rules:
  - type: "completeness"
    required_fields: ["content", "metadata"]
    threshold: 0.95
    
  - type: "uniqueness"
    key_fields: ["record_id"]
    
  - type: "timeliness"
    timestamp_field: "updated_at"
    max_age_hours: 24

alerts:
  error_rate_threshold: 0.05
  success_rate_threshold: 0.95
  max_latency_seconds: 3600
```

## 🧪 Testing

Run tests:
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=. --cov-report=html

# Run specific test file
pytest tests/test_ingestion.py

# Run with verbose output
pytest -v
```

## 📊 Performance Optimization

### Embedding Generation
- **Batch Processing**: Process texts in batches for 10x speedup
- **Caching**: 70-90% cache hit rate reduces API costs significantly
- **Parallel Processing**: Use multiple workers for large datasets

### Vector Search
- **Index Optimization**: Use appropriate index type (HNSW, IVF)
- **Quantization**: Reduce memory footprint with product quantization
- **Filtering**: Apply metadata filters before similarity search

### Pipeline Throughput
- **Incremental Processing**: Use checkpoints to avoid reprocessing
- **Parallel Connectors**: Run multiple ingestion sources concurrently
- **Async Operations**: Use async I/O for network-bound operations

## 💰 Cost Management

Track and optimize costs:
```python
from monitoring.pipeline_monitor import PipelineMetrics

metrics = PipelineMetrics()

# Embedding costs (OpenAI text-embedding-3-small: $0.00002/1K tokens)
tokens_processed = 1_000_000
embedding_cost = (tokens_processed / 1000) * 0.00002

# LLM costs (Claude Sonnet: $3/1M input, $15/1M output)
input_tokens = 500_000
output_tokens = 100_000
llm_cost = (input_tokens / 1_000_000) * 3 + (output_tokens / 1_000_000) * 15

print(f"Total Cost: ${embedding_cost + llm_cost:.2f}")
```

## 🔐 Security Considerations

- API keys stored in `.env` (never commit this file)
- Input validation on all user-provided data
- SQL injection prevention in database connectors
- Rate limiting to prevent abuse
- Secure credential management for cloud services

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **ChromaDB** for vector storage
- **Pinecone** for scalable vector search
- **Weaviate** for production vector database
- **spaCy** for NLP and entity extraction
- **Apache Airflow** for workflow orchestration
- **Neo4j** for graph database integration

## 📧 Support

For questions or issues, please open an issue on GitHub or contact the maintainers.

## 🗺️ Roadmap

- [ ] Add support for more vector databases (Qdrant, Milvus)
- [ ] Implement real-time streaming ingestion
- [ ] Add multi-modal embedding support
- [ ] Create web UI for pipeline monitoring
- [ ] Add automatic index optimization
- [ ] Implement federated learning for embeddings
- [ ] Add support for graph neural networks
- [ ] Create Kubernetes deployment templates

## 📖 Documentation

Full documentation is available in the `docs/` directory:

- [Architecture Guide](docs/architecture.md) - System architecture and design patterns
- [Pipeline Design](docs/pipeline_design.md) - Data pipeline best practices
- [Vector Databases](docs/vector_databases.md) - Choosing and configuring vector stores
- [Monitoring Guide](docs/monitoring_guide.md) - Setting up monitoring and alerts

## 🎯 Use Cases

### Enterprise Search
Build semantic search over company documents, wikis, and knowledge bases.

### Customer Support
Create intelligent support systems with document retrieval and knowledge graphs.

### Research & Discovery
Enable researchers to find relevant papers and connect related concepts.

### Content Recommendation
Power content recommendation engines with hybrid retrieval.

### Compliance & Audit
Track data lineage and ensure quality standards across pipelines.

---

**Repository**: https://github.com/ava-orange-education/Mastering-Context-Aware-AI-Engineering-with-LangChain-and-MCP/tree/main/chapter-6-data-engineering

**Documentation**: [Full API Reference](docs/api_reference.md)

**Examples**: [Complete Example Collection](examples/)