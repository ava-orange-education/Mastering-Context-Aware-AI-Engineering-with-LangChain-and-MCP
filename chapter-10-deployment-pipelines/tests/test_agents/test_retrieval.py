"""
Tests for retrieval agent
"""

import pytest
from unittest.mock import Mock, AsyncMock
from agents.retrieval_agent import RetrievalAgent
from vector_stores.base_store import Document, SearchResult


@pytest.fixture
def mock_vector_store():
    """Mock vector store"""
    store = Mock()
    store.health_check = AsyncMock(return_value=True)
    store.search = AsyncMock(return_value=[
        SearchResult(
            document=Document(
                id="doc1",
                content="Machine learning is a subset of AI",
                metadata={"source": "test"}
            ),
            score=0.95
        )
    ])
    return store


@pytest.mark.asyncio
class TestRetrievalAgent:
    """Test retrieval agent"""
    
    async def test_initialization(self, mock_vector_store):
        """Test agent initialization"""
        agent = RetrievalAgent(mock_vector_store)
        
        assert agent.vector_store == mock_vector_store
        assert agent.client is not None
    
    async def test_retrieve_documents(self, mock_vector_store):
        """Test document retrieval"""
        agent = RetrievalAgent(mock_vector_store)
        
        results = await agent.retrieve(
            query="What is machine learning?",
            top_k=5
        )
        
        assert len(results) > 0
        assert isinstance(results[0], SearchResult)
        assert mock_vector_store.search.called
    
    async def test_retrieve_with_filter(self, mock_vector_store):
        """Test retrieval with metadata filter"""
        agent = RetrievalAgent(mock_vector_store)
        
        filter_dict = {"source": "test"}
        
        await agent.retrieve(
            query="test query",
            top_k=3,
            filter=filter_dict
        )
        
        # Verify filter was passed
        call_args = mock_vector_store.search.call_args
        assert call_args.kwargs.get('filter') == filter_dict
    
    async def test_health_check(self, mock_vector_store):
        """Test health check"""
        agent = RetrievalAgent(mock_vector_store)
        
        is_healthy = await agent.health_check()
        
        assert is_healthy is True
        assert mock_vector_store.health_check.called
    
    async def test_health_check_failure(self, mock_vector_store):
        """Test health check when vector store is unhealthy"""
        mock_vector_store.health_check = AsyncMock(return_value=False)
        
        agent = RetrievalAgent(mock_vector_store)
        is_healthy = await agent.health_check()
        
        assert is_healthy is False