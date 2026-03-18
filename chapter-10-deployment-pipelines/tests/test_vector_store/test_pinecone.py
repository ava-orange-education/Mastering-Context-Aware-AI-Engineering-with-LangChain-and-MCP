"""
Tests for Pinecone vector store
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from vector_stores.pinecone_store import PineconeStore
from vector_stores.base_store import Document


@pytest.fixture
def mock_pinecone():
    """Mock Pinecone module"""
    with patch('vector_stores.pinecone_store.pinecone') as mock:
        mock.list_indexes.return_value = []
        yield mock


@pytest.mark.asyncio
class TestPineconeStore:
    """Test Pinecone vector store"""
    
    async def test_initialization(self, mock_pinecone):
        """Test store initialization"""
        store = PineconeStore()
        await store.initialize()
        
        assert store._initialized is True
        assert mock_pinecone.init.called
    
    async def test_create_index(self, mock_pinecone):
        """Test index creation"""
        store = PineconeStore()
        await store.initialize()
        
        await store.create_index(dimension=1536)
        
        assert mock_pinecone.create_index.called
    
    async def test_upsert_documents(self, mock_pinecone):
        """Test document upsert"""
        mock_index = Mock()
        mock_pinecone.Index.return_value = mock_index
        
        store = PineconeStore()
        await store.initialize()
        store.index = mock_index
        
        documents = [
            Document(
                id="doc1",
                content="Test content",
                embedding=[0.1] * 1536,
                metadata={"source": "test"}
            )
        ]
        
        await store.upsert(documents)
        
        assert mock_index.upsert.called
    
    async def test_search(self, mock_pinecone):
        """Test document search"""
        mock_index = Mock()
        mock_result = Mock()
        mock_result.matches = [
            Mock(
                id="doc1",
                score=0.95,
                metadata={"content": "Test content"}
            )
        ]
        mock_index.query.return_value = mock_result
        mock_pinecone.Index.return_value = mock_index
        
        store = PineconeStore()
        await store.initialize()
        store.index = mock_index
        
        results = await store.search(
            query_embedding=[0.1] * 1536,
            top_k=5
        )
        
        assert len(results) > 0
        assert mock_index.query.called