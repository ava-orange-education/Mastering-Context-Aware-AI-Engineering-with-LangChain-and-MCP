"""
Tests for authentication middleware
"""

import pytest
from datetime import timedelta
from api.middleware.auth_middleware import AuthMiddleware


class TestAuthMiddleware:
    """Test authentication middleware"""
    
    def test_create_access_token(self):
        """Test JWT token creation"""
        data = {"sub": "user123", "role": "user"}
        token = AuthMiddleware.create_access_token(data)
        
        assert token is not None
        assert isinstance(token, str)
        assert len(token) > 0
    
    def test_create_token_with_expiration(self):
        """Test token creation with custom expiration"""
        data = {"sub": "user123"}
        expires_delta = timedelta(minutes=15)
        
        token = AuthMiddleware.create_access_token(data, expires_delta)
        
        assert token is not None
    
    def test_verify_valid_token(self):
        """Test verifying a valid token"""
        data = {"sub": "user123", "role": "admin"}
        token = AuthMiddleware.create_access_token(data)
        
        payload = AuthMiddleware.verify_token(token)
        
        assert payload is not None
        assert payload["sub"] == "user123"
        assert payload["role"] == "admin"
    
    def test_verify_invalid_token(self):
        """Test verifying an invalid token"""
        from fastapi import HTTPException
        
        invalid_token = "invalid.token.here"
        
        with pytest.raises(HTTPException) as exc_info:
            AuthMiddleware.verify_token(invalid_token)
        
        assert exc_info.value.status_code == 401
    
    def test_verify_expired_token(self):
        """Test verifying an expired token"""
        from fastapi import HTTPException
        
        data = {"sub": "user123"}
        # Create token that expires immediately
        expires_delta = timedelta(seconds=-1)
        token = AuthMiddleware.create_access_token(data, expires_delta)
        
        with pytest.raises(HTTPException) as exc_info:
            AuthMiddleware.verify_token(token)
        
        assert exc_info.value.status_code == 401