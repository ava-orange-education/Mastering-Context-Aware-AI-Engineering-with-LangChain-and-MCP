"""
Rate limiting middleware using SlowAPI
"""

from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from fastapi import Request, Response
from fastapi.responses import JSONResponse
import logging

from config.settings import settings

logger = logging.getLogger(__name__)


# Initialize limiter
limiter = Limiter(
    key_func=get_remote_address,
    enabled=settings.rate_limit_enabled
)


def rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded) -> Response:
    """Custom handler for rate limit exceeded"""
    logger.warning(
        f"Rate limit exceeded for {get_remote_address(request)}: {exc.detail}"
    )
    
    return JSONResponse(
        status_code=429,
        content={
            "status": "error",
            "error_type": "RateLimitExceeded",
            "error_message": "Too many requests. Please try again later.",
            "detail": {
                "retry_after": exc.detail
            }
        }
    )


class RateLimitConfig:
    """Rate limit configurations for different endpoints"""
    
    # General API endpoints
    DEFAULT = f"{settings.rate_limit_per_minute}/minute"
    
    # Query endpoints (more restrictive)
    QUERY = "30/minute"
    
    # Upload endpoints (very restrictive)
    UPLOAD = "10/minute"
    
    # Health check (very permissive)
    HEALTH = "100/minute"
    
    # Authentication (moderate)
    AUTH = "20/minute"