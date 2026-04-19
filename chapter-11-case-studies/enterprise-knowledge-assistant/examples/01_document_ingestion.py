"""
Example 1: Document Ingestion

Demonstrates ingesting documents from various sources into the knowledge base
"""

import asyncio
import sys
sys.path.append('../..')

from ingestion.document_parser import DocumentParser
from ingestion.metadata_extractor import MetadataExtractor
from ingestion.chunking_strategy import ChunkingStrategy
from ingestion.embedding_pipeline import EmbeddingPipeline
from rag.permission_aware_retriever import PermissionAwareRetriever
from access_control.permission_manager import PermissionManager
from shared.base_rag import Document
from shared.utils import setup_logging

setup_logging()


async def ingest_single_document():
    """Ingest a single document"""
    
    print("\n" + "="*70)
    print("Document Ingestion - Single Document")
    print("="*70)
    
    # Parse document
    print("\n1. Parsing document...")
    parser = DocumentParser()
    
    # Simulate document (in production, use actual file)
    from pathlib import Path
    import tempfile
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("""
Enterprise Knowledge Management Strategy

Introduction
This document outlines our enterprise knowledge management strategy for 2024.

Key Objectives
1. Improve document discoverability
2. Enhance cross-team collaboration
3. Maintain security and compliance

Implementation Plan
We will deploy AI-powered search across SharePoint, Confluence, and Slack.

Contact: strategy@company.com
""")
        temp_path = f.name
    
    parsed_doc = parser.parse_file(temp_path)
    print(f"   ✅ Parsed document: {parsed_doc.file_type}")
    print(f"   Content length: {len(parsed_doc.content)} characters")
    
    # Extract metadata
    print("\n2. Extracting metadata...")
    metadata_extractor = MetadataExtractor()
    metadata = metadata_extractor.extract(
        parsed_doc,
        source="local",
        additional_metadata={
            "department": "engineering",
            "author": "strategy@company.com"
        }
    )
    print(f"   ✅ Extracted metadata")
    print(f"   Title: {metadata.get('title')}")
    print(f"   Document type: {metadata.get('document_type')}")
    print(f"   Topics: {', '.join(metadata.get('topics', [])[:5])}")
    
    # Chunk document
    print("\n3. Chunking document...")
    chunker = ChunkingStrategy(
        chunk_size=500,
        overlap=100,
        strategy="semantic"
    )
    
    chunks = chunker.chunk_document(
        content=parsed_doc.content,
        metadata=metadata,
        document_id="doc_001"
    )
    print(f"   ✅ Created {len(chunks)} chunks")
    
    # Generate embeddings
    print("\n4. Generating embeddings...")
    embedder = EmbeddingPipeline()
    chunks_with_embeddings = await embedder.process_chunks(chunks)
    print(f"   ✅ Generated embeddings for {len(chunks_with_embeddings)} chunks")
    
    # Convert to Documents
    print("\n5. Converting to Document objects...")
    documents = []
    for chunk in chunks_with_embeddings:
        doc = Document(
            id=chunk.chunk_id,
            content=chunk.text,
            metadata=chunk.metadata,
            embedding=chunk.embedding
        )
        documents.append(doc)
    print(f"   ✅ Created {len(documents)} Document objects")
    
    # Set permissions
    print("\n6. Setting permissions...")
    permission_manager = PermissionManager()
    
    for doc in documents:
        await permission_manager.set_permissions(
            document_id=doc.id,
            permissions={
                "public": False,
                "owner": "strategy@company.com",
                "groups": ["engineering", "product"],
                "departments": ["engineering"]
            }
        )
    print(f"   ✅ Set permissions for {len(documents)} documents")
    
    # Store in vector database
    print("\n7. Storing in vector database...")
    retriever = PermissionAwareRetriever()
    await retriever.initialize()
    await retriever.upsert(documents)
    print(f"   ✅ Stored {len(documents)} documents in vector database")
    
    # Cleanup
    Path(temp_path).unlink()
    
    print("\n" + "="*70)
    print("Ingestion Complete!")
    print("="*70)
    print(f"✓ Document ID: doc_001")
    print(f"✓ Chunks created: {len(chunks)}")
    print(f"✓ Permissions set: engineering, product")


async def ingest_batch_documents():
    """Ingest multiple documents in batch"""
    
    print("\n" + "="*70)
    print("Document Ingestion - Batch Processing")
    print("="*70)
    
    # Simulate multiple documents
    documents_data = [
        {
            "id": f"doc_{i:03d}",
            "content": f"Document {i} content about project planning and execution.",
            "metadata": {
                "title": f"Project Document {i}",
                "department": "engineering" if i % 2 == 0 else "product",
                "author": f"user{i}@company.com",
                "source": "confluence"
            }
        }
        for i in range(1, 11)
    ]
    
    print(f"\n1. Processing {len(documents_data)} documents...")
    
    # Initialize components
    metadata_extractor = MetadataExtractor()
    chunker = ChunkingStrategy(chunk_size=500, strategy="semantic")
    embedder = EmbeddingPipeline()
    retriever = PermissionAwareRetriever()
    permission_manager = PermissionManager()
    
    await retriever.initialize()
    
    all_documents = []
    
    for doc_data in documents_data:
        # Extract metadata
        from ingestion.document_parser import ParsedDocument
        parsed = ParsedDocument(
            content=doc_data["content"],
            metadata=doc_data["metadata"],
            file_type=".txt"
        )
        
        metadata = metadata_extractor.extract(parsed, source="confluence")
        
        # Chunk
        chunks = chunker.chunk_document(
            content=doc_data["content"],
            metadata=metadata,
            document_id=doc_data["id"]
        )
        
        # Generate embeddings
        chunks_with_embeddings = await embedder.process_chunks(chunks)
        
        # Convert to Documents
        for chunk in chunks_with_embeddings:
            doc = Document(
                id=chunk.chunk_id,
                content=chunk.text,
                metadata=chunk.metadata,
                embedding=chunk.embedding
            )
            all_documents.append(doc)
            
            # Set permissions
            await permission_manager.set_permissions(
                document_id=doc.id,
                permissions={
                    "public": False,
                    "groups": [metadata["department"]],
                    "departments": [metadata["department"]]
                }
            )
    
    print(f"   ✅ Processed {len(documents_data)} documents")
    print(f"   ✅ Created {len(all_documents)} chunks total")
    
    # Batch upsert
    print("\n2. Batch uploading to vector database...")
    await retriever.upsert(all_documents)
    print(f"   ✅ Uploaded {len(all_documents)} document chunks")
    
    print("\n" + "="*70)
    print("Batch Ingestion Complete!")
    print("="*70)
    print(f"✓ Documents processed: {len(documents_data)}")
    print(f"✓ Total chunks: {len(all_documents)}")
    print(f"✓ Permissions configured for all documents")


async def main():
    """Run all examples"""
    
    await ingest_single_document()
    
    await ingest_batch_documents()
    
    print("\n" + "="*70)
    print("All Ingestion Examples Complete!")
    print("="*70)


if __name__ == "__main__":
    asyncio.run(main())