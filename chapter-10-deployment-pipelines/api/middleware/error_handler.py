"""
Global error handling middleware
"""

from fastapi import Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError
import logging
import traceback
from datetime import datetime

from api.models.response_models import ErrorResponse, ResponseStatus

logger = logging.getLogger(__name__)


async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError
) -> JSONResponse:
    """Handle Pydantic validation errors"""
    logger.warning(f"Validation error: {exc.errors()}")
    
    error_response = ErrorResponse(
        status=ResponseStatus.ERROR,
        error_type="ValidationError",
        error_message="Request validation failed",
        detail={
            "errors": exc.errors(),
            "body": str(exc.body) if hasattr(exc, 'body') else None
        }
    )
    
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=error_response.model_dump()
    )


async def general_exception_handler(
    request: Request,
    exc: Exception
) -> JSONResponse:
    """Handle general exceptions"""
    logger.error(
        f"Unhandled exception: {type(exc).__name__}: {str(exc)}\n"
        f"Traceback: {traceback.format_exc()}"
    )
    
    error_response = ErrorResponse(
        status=ResponseStatus.ERROR,
        error_type=type(exc).__name__,
        error_message="An internal error occurred",
        detail={
            "message": str(exc) if not isinstance(exc, Exception) else None
        }
    )
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=error_response.model_dump()
    )


class ErrorHandler:
    """Centralized error handling utilities"""
    
    @staticmethod
    def create_error_response(
        error_type: str,
        error_message: str,
        status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail: dict = None
    ) -> JSONResponse:
        """Create standardized error response"""
        error_response = ErrorResponse(
            error_type=error_type,
            error_message=error_message,
            detail=detail or {}
        )
        
        return JSONResponse(
            status_code=status_code,
            content=error_response.model_dump()
        )
    
    @staticmethod
    def log_error(
        error_type: str,
        error_message: str,
        context: dict = None
    ):
        """Log error with context"""
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "error_type": error_type,
            "error_message": error_message,
            "context": context or {}
        }
        
        logger.error(f"Error: {log_data}")