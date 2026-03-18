"""
Middleware module
"""

from .auth_middleware import (
    AuthMiddleware,
    require_auth,
    optional_auth,
    security
)

from .rate_limit_middleware import (
    limiter,
    rate_limit_exceeded_handler,
    RateLimitConfig
)

from .error_handler import (
    validation_exception_handler,
    general_exception_handler,
    ErrorHandler
)

__all__ = [
    # Auth
    'AuthMiddleware',
    'require_auth',
    'optional_auth',
    'security',
    
    # Rate limiting
    'limiter',
    'rate_limit_exceeded_handler',
    'RateLimitConfig',
    
    # Error handling
    'validation_exception_handler',
    'general_exception_handler',
    'ErrorHandler',
]