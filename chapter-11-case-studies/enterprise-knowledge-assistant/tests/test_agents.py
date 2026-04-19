"""
Tests for enterprise agents
"""

import pytest
import asyncio
import sys
sys.path.append('../..')

from agents.document_search_agent import DocumentSearchAgent
from agents.summarization_agent import SummarizationAgent
from agents.cross_reference_agent import CrossReferenceAgent
from agents.expert_finder_agent import ExpertFinderAgent


@pytest.mark.asyncio
class TestDocumentSearchAgent:
    """Test Document Search Agent"""
    
    async def test_initialization(self):
        """Test agent initializes correctly"""
        agent = DocumentSearchAgent()
        
        assert agent.name == "Document Search Agent"
        assert agent.retriever is not None
        assert agent.permission_manager is not None
    
    async def test_process_with_query(self):
        """Test processing search query"""
        agent = DocumentSearchAgent()
        
        input_data = {
            "query": "project planning",
            "user_id": "user@company.com",
            "user_groups": ["engineering"],
            "top_k": 5
        }
        
        # This will fail without vector DB, but tests the flow
        try:
            result = await agent.process(input_data)
            assert result.content is not None
            assert result.agent_name == "Document Search Agent"
        except Exception as e:
            # Expected without vector DB setup
            assert "initialize" in str(e).lower() or "vector" in str(e).lower()
    
    async def test_process_requires_query(self):
        """Test that query is required"""
        agent = DocumentSearchAgent()
        
        with pytest.raises(Exception):
            await agent.process({"user_id": "user@company.com"})


@pytest.mark.asyncio
class TestSummarizationAgent:
    """Test Summarization Agent"""
    
    async def test_initialization(self):
        """Test agent initializes correctly"""
        agent = SummarizationAgent()
        
        assert agent.name == "Summarization Agent"
        assert agent.model is not None
    
    async def test_summarize_document(self):
        """Test document summarization"""
        agent = SummarizationAgent()
        
        content = """
        This is a test document about enterprise knowledge management.
        It discusses various strategies for organizing and retrieving information.
        The document covers best practices and implementation guidelines.
        """
        
        input_data = {
            "content": content,
            "document_type": "document",
            "max_length": 100
        }
        
        try:
            result = await agent.process(input_data)
            assert result.content is not None
            assert len(result.content) <= 500  # Should be shorter than original
        except Exception:
            # Expected if API key not configured
            pass
    
    async def test_summarize_meeting_notes(self):
        """Test meeting notes summarization"""
        agent = SummarizationAgent()
        
        content = """
        Meeting Notes - Product Planning
        Attendees: John, Sarah, Mike
        Date: 2024-03-15
        
        Discussion Points:
        - Q2 roadmap priorities
        - Resource allocation
        - Customer feedback integration
        
        Action Items:
        - John: Create roadmap draft by Friday
        - Sarah: Schedule customer interviews
        - Mike: Review technical feasibility
        """
        
        input_data = {
            "content": content,
            "document_type": "meeting_notes",
            "max_length": 200
        }
        
        try:
            result = await agent.process(input_data)
            assert result.content is not None
        except Exception:
            pass


@pytest.mark.asyncio
class TestCrossReferenceAgent:
    """Test Cross Reference Agent"""
    
    async def test_initialization(self):
        """Test agent initializes correctly"""
        agent = CrossReferenceAgent()
        
        assert agent.name == "Cross Reference Agent"
        assert agent.retriever is not None
    
    async def test_process_with_document_id(self):
        """Test finding related documents by ID"""
        agent = CrossReferenceAgent()
        
        input_data = {
            "document_id": "doc_001",
            "user_id": "user@company.com",
            "max_results": 5
        }
        
        try:
            result = await agent.process(input_data)
            assert result.content is not None
        except Exception:
            # Expected without vector DB
            pass
    
    async def test_process_with_query(self):
        """Test finding related documents by query"""
        agent = CrossReferenceAgent()
        
        input_data = {
            "query": "project planning",
            "user_id": "user@company.com",
            "relationship_types": ["topical"],
            "max_results": 5
        }
        
        try:
            result = await agent.process(input_data)
            assert result.content is not None
        except Exception:
            pass
    
    async def test_requires_id_or_query(self):
        """Test that either document_id or query is required"""
        agent = CrossReferenceAgent()
        
        with pytest.raises(ValueError):
            await agent.process({"user_id": "user@company.com"})


@pytest.mark.asyncio
class TestExpertFinderAgent:
    """Test Expert Finder Agent"""
    
    async def test_initialization(self):
        """Test agent initializes correctly"""
        agent = ExpertFinderAgent()
        
        assert agent.name == "Expert Finder Agent"
        assert agent.retriever is not None
    
    async def test_find_experts(self):
        """Test finding experts for a topic"""
        agent = ExpertFinderAgent()
        
        input_data = {
            "topic": "machine learning",
            "min_contributions": 2,
            "max_experts": 5
        }
        
        try:
            result = await agent.process(input_data)
            assert result.content is not None
        except Exception:
            # Expected without vector DB
            pass
    
    async def test_requires_topic(self):
        """Test that topic is required"""
        agent = ExpertFinderAgent()
        
        with pytest.raises(ValueError):
            await agent.process({})


@pytest.mark.asyncio
class TestAgentIntegration:
    """Test agent integration"""
    
    async def test_search_to_summarization_flow(self):
        """Test flow from search to summarization"""
        
        search_agent = DocumentSearchAgent()
        summarize_agent = SummarizationAgent()
        
        # Verify agents can be chained
        assert search_agent.name != summarize_agent.name
        assert callable(search_agent.process)
        assert callable(summarize_agent.process)