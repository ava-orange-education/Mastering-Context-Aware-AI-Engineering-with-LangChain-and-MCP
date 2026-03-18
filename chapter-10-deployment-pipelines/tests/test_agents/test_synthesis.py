"""
Tests for synthesis agent
"""

import pytest
from unittest.mock import Mock, patch
from agents.synthesis_agent import SynthesisAgent


@pytest.fixture
def sample_analysis():
    """Sample analysis for testing"""
    return {
        "analysis": "The documents discuss machine learning and AI concepts.",
        "document_count": 3,
        "processing_time": 1.2
    }


@pytest.mark.asyncio
class TestSynthesisAgent:
    """Test synthesis agent"""
    
    async def test_initialization(self):
        """Test agent initialization"""
        agent = SynthesisAgent()
        
        assert agent.client is not None
        assert agent.model is not None
    
    async def test_create_synthesis_prompt(self, sample_analysis):
        """Test synthesis prompt creation"""
        agent = SynthesisAgent()
        
        query = "What is AI?"
        context = "AI is artificial intelligence..."
        
        prompt = agent._create_synthesis_prompt(
            query,
            sample_analysis["analysis"],
            context
        )
        
        assert query in prompt
        assert sample_analysis["analysis"] in prompt
        assert context in prompt
        assert "synthesize" in prompt.lower() or "response" in prompt.lower()
    
    @patch('agents.synthesis_agent.anthropic.Anthropic')
    async def test_synthesize(self, mock_anthropic, sample_analysis):
        """Test response synthesis"""
        # Mock the Anthropic client
        mock_client = Mock()
        mock_response = Mock()
        mock_response.content = [Mock(text="AI is artificial intelligence...")]
        mock_response.usage = Mock(input_tokens=100, output_tokens=50)
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.return_value = mock_client
        
        agent = SynthesisAgent()
        agent.client = mock_client
        
        result = await agent.synthesize(
            query="What is AI?",
            analysis=sample_analysis,
            context="Context about AI"
        )
        
        assert "response" in result
        assert "processing_time" in result
        assert "input_tokens" in result
        assert "output_tokens" in result
        assert isinstance(result["response"], str)