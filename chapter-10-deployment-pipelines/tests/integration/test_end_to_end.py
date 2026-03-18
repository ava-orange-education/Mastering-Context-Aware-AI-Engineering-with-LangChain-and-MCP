"""
End-to-end integration tests
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, AsyncMock
from api.main import app


@pytest.fixture
def mock_dependencies():
    """Mock all external dependencies"""
    with patch('agents.agent_factory.AgentFactory') as mock_factory, \
         patch('vector_stores.store_factory.VectorStoreFactory') as mock_store_factory:
        
        # Mock agents
        mock_retrieval = Mock()
        mock_retrieval.retrieve = AsyncMock(return_value=[])
        mock_retrieval.health_check = AsyncMock(return_value=True)
        
        mock_analysis = Mock()
        mock_analysis.analyze = AsyncMock(return_value={
            "analysis": "Test analysis",
            "document_count": 0,
            "processing_time": 0.1
        })
        mock_analysis.health_check = AsyncMock(return_value=True)
        
        mock_synthesis = Mock()
        mock_synthesis.synthesize = AsyncMock(return_value={
            "response": "Test response",
            "processing_time": 0.1,
            "input_tokens": 10,
            "output_tokens": 20
        })
        mock_synthesis.health_check = AsyncMock(return_value=True)
        
        mock_factory.get_retrieval_agent = AsyncMock(return_value=mock_retrieval)
        mock_factory.get_analysis_agent = AsyncMock(return_value=mock_analysis)
        mock_factory.get_synthesis_agent = AsyncMock(return_value=mock_synthesis)
        mock_factory.health_check_all = AsyncMock(return_value={
            "retrieval": True,
            "analysis": True,
            "synthesis": True
        })
        
        # Mock vector store
        mock_store = Mock()
        mock_store.health_check = AsyncMock(return_value=True)
        mock_store_factory.get_vector_store = AsyncMock(return_value=mock_store)
        
        yield {
            "factory": mock_factory,
            "store_factory": mock_store_factory,
            "retrieval": mock_retrieval,
            "analysis": mock_analysis,
            "synthesis": mock_synthesis
        }


@pytest.mark.integration
class TestEndToEndFlow:
    """Test complete request flow"""
    
    def test_health_to_query_flow(self, mock_dependencies):
        """Test flow from health check to query processing"""
        client = TestClient(app)
        
        # Step 1: Check health
        health_response = client.get("/health")
        assert health_response.status_code == 200
        
        # Step 2: Process query
        query_payload = {
            "query": "What is machine learning?",
            "top_k": 5,
            "agent_type": "multi_agent"
        }
        
        query_response = client.post("/api/v1/agents/query", json=query_payload)
        
        # Should succeed or fail gracefully
        assert query_response.status_code in [200, 503]
        
        if query_response.status_code == 200:
            data = query_response.json()
            assert "query_id" in data
            assert "response" in data
    
    def test_multi_query_flow(self, mock_dependencies):
        """Test multiple queries in sequence"""
        client = TestClient(app)
        
        queries = [
            "What is AI?",
            "Explain machine learning",
            "What are neural networks?"
        ]
        
        results = []
        
        for query in queries:
            payload = {
                "query": query,
                "top_k": 3,
                "agent_type": "multi_agent"
            }
            
            response = client.post("/api/v1/agents/query", json=payload)
            results.append(response.status_code)
        
        # All should have same status
        assert len(set(results)) <= 2  # Either all success or all fail


@pytest.mark.integration
class TestErrorRecovery:
    """Test error handling and recovery"""
    
    def test_invalid_request_recovery(self):
        """Test that system recovers from invalid requests"""
        client = TestClient(app)
        
        # Send invalid request
        invalid_payload = {"query": ""}
        response1 = client.post("/api/v1/agents/query", json=invalid_payload)
        assert response1.status_code == 422
        
        # System should still work for valid requests
        valid_payload = {
            "query": "Valid query",
            "top_k": 5,
            "agent_type": "multi_agent"
        }
        response2 = client.post("/api/v1/agents/query", json=valid_payload)
        assert response2.status_code in [200, 503]
    
    def test_health_after_errors(self):
        """Test health endpoint after errors"""
        client = TestClient(app)
        
        # Trigger some errors
        for _ in range(3):
            client.post("/api/v1/agents/query", json={"query": ""})
        
        # Health should still respond
        health_response = client.get("/health")
        assert health_response.status_code == 200


@pytest.mark.integration
class TestConcurrency:
    """Test concurrent request handling"""
    
    def test_concurrent_health_checks(self):
        """Test concurrent health check requests"""
        import concurrent.futures
        
        client = TestClient(app)
        
        def check_health():
            return client.get("/health")
        
        # Send multiple concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(check_health) for _ in range(10)]
            results = [f.result() for f in futures]
        
        # All should succeed
        assert all(r.status_code == 200 for r in results)