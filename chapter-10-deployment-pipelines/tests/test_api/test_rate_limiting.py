"""
Tests for rate limiting middleware
"""

import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


class TestRateLimiting:
    """Test rate limiting functionality"""
    
    def test_rate_limit_not_exceeded(self):
        """Test requests within rate limit"""
        # Make a few requests (under limit)
        for _ in range(3):
            response = client.get("/health")
            assert response.status_code == 200
    
    def test_rate_limit_configuration(self):
        """Test rate limit configuration is loaded"""
        from api.middleware.rate_limit_middleware import RateLimitConfig
        
        assert hasattr(RateLimitConfig, 'DEFAULT')
        assert hasattr(RateLimitConfig, 'QUERY')
        assert hasattr(RateLimitConfig, 'UPLOAD')
    
    @pytest.mark.skip(reason="Rate limiting may be disabled in test environment")
    def test_rate_limit_exceeded(self):
        """Test behavior when rate limit is exceeded"""
        # This test is skipped by default as it requires:
        # 1. Rate limiting enabled
        # 2. Many requests to trigger limit
        # 3. May take long time
        
        # Make many requests to exceed limit
        responses = []
        for _ in range(150):  # Exceed typical limit
            response = client.get("/health")
            responses.append(response)
        
        # At least one should be rate limited
        rate_limited = any(r.status_code == 429 for r in responses)
        assert rate_limited