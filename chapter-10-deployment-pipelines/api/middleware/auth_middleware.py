"""
Authentication middleware for FastAPI
"""

from fastapi import Request, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from datetime import datetime, timedelta
from typing import Optional, Dict
import logging

from config.settings import settings

logger = logging.getLogger(__name__)

security = HTTPBearer()


class AuthMiddleware:
    """JWT-based authentication middleware"""
    
    @staticmethod
    def create_access_token(data: Dict, expires_delta: Optional[timedelta] = None) -> str:
        """Create JWT access token"""
        to_encode = data.copy()
        
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(
                minutes=settings.access_token_expire_minutes
            )
        
        to_encode.update({"exp": expire})
        
        encoded_jwt = jwt.encode(
            to_encode,
            settings.jwt_secret_key,
            algorithm=settings.jwt_algorithm
        )
        
        return encoded_jwt
    
    @staticmethod
    def verify_token(token: str) -> Dict:
        """Verify and decode JWT token"""
        try:
            payload = jwt.decode(
                token,
                settings.jwt_secret_key,
                algorithms=[settings.jwt_algorithm]
            )
            return payload
        
        except JWTError as e:
            logger.error(f"JWT verification failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Could not validate credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
    
    @staticmethod
    async def get_current_user(
        credentials: HTTPAuthorizationCredentials = security
    ) -> Dict:
        """Extract current user from JWT token"""
        token = credentials.credentials
        payload = AuthMiddleware.verify_token(token)
        
        user_id = payload.get("sub")
        if user_id is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials"
            )
        
        return {"user_id": user_id, "payload": payload}


# Dependency for routes requiring authentication
async def require_auth(
    credentials: HTTPAuthorizationCredentials = security
) -> Dict:
    """FastAPI dependency for authenticated routes"""
    return await AuthMiddleware.get_current_user(credentials)


# Optional authentication (doesn't fail if no token)
async def optional_auth(request: Request) -> Optional[Dict]:
    """FastAPI dependency for optionally authenticated routes"""
    auth_header = request.headers.get("Authorization")
    
    if not auth_header or not auth_header.startswith("Bearer "):
        return None
    
    token = auth_header.split(" ")[1]
    
    try:
        return AuthMiddleware.verify_token(token)
    except HTTPException:
        return None