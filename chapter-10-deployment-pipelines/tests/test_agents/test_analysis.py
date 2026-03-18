"""
Tests for analysis agent
"""

import pytest
from unittest.mock import Mock, patch
from agents.analysis_agent import AnalysisAgent
from vector_stores.base_store import Document, SearchResult


@pytest.fixture
def sample_search_results():
    """Sample search results for testing"""
    return [
        SearchResult(
            document=Document(
                id="doc1",
                content="Machine learning is a subset of AI that enables systems to learn from data.",
                metadata={"source": "textbook"}
            ),
            score=0.95
        ),
        SearchResult(
            document=Document(
                id="doc2",
                content="Deep learning uses neural networks with multiple layers.",
                metadata={"source": "article"}
            ),
            score=0.88
        )
    ]


@pytest.mark.asyncio
class TestAnalysisAgent:
    """Test analysis agent"""
    
    async def test_initialization(self):
        """Test agent initialization"""
        agent = AnalysisAgent()
        
        assert agent.client is not None
        assert agent.model is not None
    
    async def test_prepare_context(self, sample_search_results):
        """Test context preparation"""
        agent = AnalysisAgent()
        
        context = agent._prepare_context(sample_search_results)
        
        assert isinstance(context, str)
        assert "Machine learning" in context
        assert "Deep learning" in context
        assert len(context) > 0
    
    async def test_create_analysis_prompt(self):
        """Test analysis prompt creation"""
        agent = AnalysisAgent()
        
        query = "What is machine learning?"
        context = "Sample context about ML"
        
        prompt = agent._create_analysis_prompt(query, context)
        
        assert query in prompt
        assert context in prompt
        assert "analyze" in prompt.lower()
    
    @patch('agents.analysis_agent.anthropic.Anthropic')
    async def test_analyze(self, mock_anthropic, sample_search_results):
        """Test document analysis"""
        # Mock the Anthropic client
        mock_client = Mock()
        mock_response = Mock()
        mock_response.content = [Mock(text="Detailed analysis of the documents")]
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.return_value = mock_client
        
        agent = AnalysisAgent()
        agent.client = mock_client
        
        result = await agent.analyze(
            query="What is AI?",
            documents=sample_search_results
        )
        
        assert "analysis" in result
        assert "document_count" in result
        assert "processing_time" in result
        assert result["document_count"] == len(sample_search_results)