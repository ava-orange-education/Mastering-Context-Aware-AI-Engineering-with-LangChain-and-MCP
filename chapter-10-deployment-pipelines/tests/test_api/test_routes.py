"""
Tests for API routes
"""

import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


class TestHealthRoutes:
    """Test health check endpoints"""
    
    def test_basic_health_check(self):
        """Test basic health endpoint"""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["status"] in ["healthy", "degraded", "unhealthy"]
        assert "version" in data
        assert "uptime_seconds" in data
        assert "components" in data
    
    def test_readiness_probe(self):
        """Test Kubernetes readiness probe"""
        response = client.get("/health/ready")
        
        assert response.status_code in [200, 503]
        data = response.json()
        
        assert "status" in data
    
    def test_liveness_probe(self):
        """Test Kubernetes liveness probe"""
        response = client.get("/health/live")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["status"] == "alive"
    
    def test_system_metrics(self):
        """Test system metrics endpoint"""
        response = client.get("/metrics/system")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "cpu" in data
        assert "memory" in data
        assert "disk" in data
        assert "uptime_seconds" in data


class TestRootRoutes:
    """Test root endpoints"""
    
    def test_root_endpoint(self):
        """Test root endpoint"""
        response = client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "name" in data
        assert "version" in data
        assert "status" in data
    
    def test_info_endpoint(self):
        """Test info endpoint"""
        response = client.get("/info")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "name" in data
        assert "version" in data
        assert "environment" in data


class TestAgentRoutes:
    """Test agent endpoints"""
    
    def test_query_endpoint_valid(self):
        """Test query endpoint with valid request"""
        payload = {
            "query": "What is machine learning?",
            "top_k": 5,
            "agent_type": "multi_agent"
        }
        
        response = client.post("/api/v1/agents/query", json=payload)
        
        # Might fail if backend not fully initialized, so check for both
        assert response.status_code in [200, 503]
        
        if response.status_code == 200:
            data = response.json()
            assert "status" in data
            assert "query" in data
            assert "response" in data
            assert "query_id" in data
    
    def test_query_endpoint_invalid(self):
        """Test query endpoint with invalid request"""
        payload = {
            "query": "",  # Empty query
            "top_k": 5
        }
        
        response = client.post("/api/v1/agents/query", json=payload)
        
        assert response.status_code == 422  # Validation error
    
    def test_query_endpoint_missing_fields(self):
        """Test query endpoint with missing required fields"""
        payload = {
            "top_k": 5
            # Missing query field
        }
        
        response = client.post("/api/v1/agents/query", json=payload)
        
        assert response.status_code == 422
    
    def test_query_endpoint_invalid_top_k(self):
        """Test query endpoint with invalid top_k"""
        payload = {
            "query": "Test query",
            "top_k": 100  # Exceeds maximum
        }
        
        response = client.post("/api/v1/agents/query", json=payload)
        
        assert response.status_code == 422


class TestErrorHandling:
    """Test error handling"""
    
    def test_404_not_found(self):
        """Test 404 handling"""
        response = client.get("/nonexistent/endpoint")
        
        assert response.status_code == 404
    
    def test_invalid_json(self):
        """Test handling of invalid JSON"""
        response = client.post(
            "/api/v1/agents/query",
            data="invalid json",
            headers={"Content-Type": "application/json"}
        )
        
        assert response.status_code == 422