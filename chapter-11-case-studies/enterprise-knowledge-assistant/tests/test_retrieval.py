"""
Tests for retrieval components
"""

import pytest
import sys
sys.path.append('../..')

from rag.permission_aware_retriever import PermissionAwareRetriever
from rag.multi_source_retriever import MultiSourceRetriever
from rag.context_merger import ContextMerger
from shared.base_rag import Document, SearchResult
from ingestion.document_parser import DocumentParser
from ingestion.metadata_extractor import MetadataExtractor
from ingestion.chunking_strategy import ChunkingStrategy, Chunk


class TestDocumentParser:
    """Test Document Parser"""
    
    def test_parse_text_file(self):
        """Test parsing text file"""
        parser = DocumentParser()
        
        # Create temporary text file
        import tempfile
        from pathlib import Path
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Sample text content")
            temp_path = f.name
        
        try:
            parsed = parser.parse_file(temp_path)
            
            assert parsed.content == "Sample text content"
            assert parsed.file_type == ".txt"
            assert "filename" in parsed.metadata
        finally:
            Path(temp_path).unlink()
    
    def test_unsupported_format(self):
        """Test unsupported file format"""
        parser = DocumentParser()
        
        with pytest.raises(ValueError):
            parser.parse_file("file.unsupported")


class TestMetadataExtractor:
    """Test Metadata Extractor"""
    
    def test_extract_metadata(self):
        """Test metadata extraction"""
        from ingestion.document_parser import ParsedDocument
        
        extractor = MetadataExtractor()
        
        parsed = ParsedDocument(
            content="# Title\n\nContent here\n\nMore content",
            metadata={"filename": "test.md"},
            file_type=".md"
        )
        
        metadata = extractor.extract(parsed, source="confluence")
        
        assert "title" in metadata
        assert "source" in metadata
        assert metadata["source"] == "confluence"
        assert "topics" in metadata
    
    def test_extract_entities(self):
        """Test entity extraction"""
        extractor = MetadataExtractor()
        
        content = "Contact user@example.com or visit https://example.com by 12/31/2024"
        
        entities = extractor._extract_entities(content)
        
        assert "emails" in entities
        assert len(entities["emails"]) > 0
        assert "urls" in entities
        assert len(entities["urls"]) > 0


class TestChunkingStrategy:
    """Test Chunking Strategy"""
    
    def test_fixed_size_chunking(self):
        """Test fixed size chunking"""
        chunker = ChunkingStrategy(
            chunk_size=100,
            overlap=20,
            strategy="fixed_size"
        )
        
        content = "A" * 500  # 500 character content
        metadata = {"doc_id": "test"}
        
        chunks = chunker.chunk_document(content, metadata, "doc_001")
        
        assert len(chunks) > 1
        assert all(isinstance(chunk, Chunk) for chunk in chunks)
    
    def test_semantic_chunking(self):
        """Test semantic chunking"""
        chunker = ChunkingStrategy(
            chunk_size=200,
            overlap=50,
            strategy="semantic"
        )
        
        content = """
# Introduction
This is the introduction section.

# Methods
This section describes the methods.

# Results
Here are the results.
"""
        
        chunks = chunker.chunk_document(content, {}, "doc_002")
        
        assert len(chunks) > 0
        # Should preserve section boundaries when possible
    
    def test_chunk_has_metadata(self):
        """Test chunks include metadata"""
        chunker = ChunkingStrategy(chunk_size=100, strategy="fixed_size")
        
        metadata = {"author": "test@example.com", "department": "engineering"}
        content = "Test content " * 50
        
        chunks = chunker.chunk_document(content, metadata, "doc_003")
        
        assert len(chunks) > 0
        assert chunks[0].metadata["author"] == "test@example.com"
        assert chunks[0].metadata["department"] == "engineering"


class TestContextMerger:
    """Test Context Merger"""
    
    def test_merge_ranked(self):
        """Test ranked merging strategy"""
        merger = ContextMerger(max_context_length=1000)
        
        # Create mock search results
        results = []
        for i in range(3):
            doc = Document(
                id=f"doc_{i}",
                content=f"Content {i} " * 20,
                metadata={"title": f"Document {i}", "source": "confluence"}
            )
            result = SearchResult(
                document=doc,
                score=1.0 - (i * 0.1),
                rank=i+1
            )
            results.append(result)
        
        context = merger.merge_contexts(results, "test query", strategy="ranked")
        
        assert len(context) > 0
        assert len(context) <= 1000
        assert "Document 0" in context  # Highest ranked
    
    def test_merge_interleaved(self):
        """Test interleaved merging strategy"""
        merger = ContextMerger(max_context_length=1000)
        
        results = []
        for i in range(3):
            doc = Document(
                id=f"doc_{i}",
                content=f"Content from source {i % 2}",
                metadata={
                    "title": f"Doc {i}",
                    "source": f"source_{i % 2}"
                }
            )
            result = SearchResult(document=doc, score=0.9, rank=i+1)
            results.append(result)
        
        context = merger.merge_contexts(results, "query", strategy="interleaved")
        
        assert len(context) > 0
    
    def test_summarize_sources(self):
        """Test source summarization"""
        merger = ContextMerger()
        
        results = []
        for i in range(5):
            doc = Document(
                id=f"doc_{i}",
                content="content",
                metadata={
                    "source": "sharepoint" if i < 3 else "confluence",
                    "department": "engineering"
                }
            )
            result = SearchResult(document=doc, score=0.8, rank=i+1)
            results.append(result)
        
        summary = merger.summarize_sources(results)
        
        assert summary["total_documents"] == 5
        assert "sharepoint" in summary["sources"]
        assert "confluence" in summary["sources"]
        assert summary["sources"]["sharepoint"] == 3
        assert summary["sources"]["confluence"] == 2


@pytest.mark.asyncio
class TestPermissionAwareRetriever:
    """Test Permission Aware Retriever"""
    
    async def test_initialization(self):
        """Test retriever initializes"""
        retriever = PermissionAwareRetriever()
        
        assert retriever.collection_name == "enterprise_documents"
        assert retriever.permission_manager is not None
    
    async def test_search_requires_user_id(self):
        """Test search requires user ID"""
        retriever = PermissionAwareRetriever()
        
        with pytest.raises(Exception):
            await retriever.search(query="test", user_id=None)


@pytest.mark.asyncio
class TestMultiSourceRetriever:
    """Test Multi-Source Retriever"""
    
    async def test_initialization(self):
        """Test retriever initializes"""
        retriever = MultiSourceRetriever()
        
        assert retriever.retriever is not None
        assert retriever.source_weights is not None
    
    async def test_merge_results(self):
        """Test result merging"""
        retriever = MultiSourceRetriever()
        
        # Create duplicate and unique results
        results = []
        for i in range(5):
            doc = Document(
                id=f"doc_{i}",
                content="content",
                metadata={"source": "confluence"}
            )
            result = SearchResult(document=doc, score=0.9 - (i * 0.1), rank=i+1)
            results.append(result)
        
        # Add duplicate
        dup_result = SearchResult(
            document=results[0].document,
            score=0.85,
            rank=6
        )
        results.append(dup_result)
        
        merged = retriever._merge_results(results, top_k=5)
        
        # Should deduplicate
        assert len(merged) <= 5
        # Should be sorted by score
        assert merged[0].score >= merged[-1].score