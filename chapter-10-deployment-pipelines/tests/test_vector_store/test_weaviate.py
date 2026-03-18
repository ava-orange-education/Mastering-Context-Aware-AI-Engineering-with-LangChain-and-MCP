"""
Tests for Weaviate vector store
"""

import pytest
from unittest.mock import Mock, patch
from vector_stores.weaviate_store import WeaviateStore
from vector_stores.base_store import Document


@pytest.fixture
def mock_weaviate():
    """Mock Weaviate module"""
    with patch('vector_stores.weaviate_store.weaviate') as mock:
        mock_client = Mock()
        mock_client.is_ready.return_value = True
        mock.Client.return_value = mock_client
        yield mock


@pytest.mark.asyncio
class TestWeaviateStore:
    """Test Weaviate vector store"""
    
    async def test_initialization(self, mock_weaviate):
        """Test store initialization"""
        store = WeaviateStore()
        await store.initialize()
        
        assert store._initialized is True
        assert store.client is not None
    
    async def test_create_index(self, mock_weaviate):
        """Test schema creation"""
        mock_client = mock_weaviate.Client.return_value
        mock_client.schema.get.return_value = {"classes": []}
        
        store = WeaviateStore()
        await store.initialize()
        
        await store.create_index(dimension=1536)
        
        assert mock_client.schema.create_class.called
    
    async def test_upsert_documents(self, mock_weaviate):
        """Test document upsert"""
        mock_client = mock_weaviate.Client.return_value
        mock_batch = Mock()
        mock_client.batch = mock_batch
        
        store = WeaviateStore()
        await store.initialize()
        
        documents = [
            Document(
                id="doc1",
                content="Test content",
                embedding=[0.1] * 1536,
                metadata={"source": "test"}
            )
        ]
        
        await store.upsert(documents)
        
        # Verify batch was used
        assert mock_batch.__enter__.called or mock_batch.__aenter__.called