"""
Error handling and retry logic for production systems.
"""

import time
import functools
from typing import Callable, Any, Optional, Type, Tuple
import logging

logger = logging.getLogger(__name__)


class RetryHandler:
    """Handle retries with exponential backoff"""
    
    @staticmethod
    def exponential_backoff(
        func: Callable,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exceptions: Tuple[Type[Exception], ...] = (Exception,)
    ) -> Callable:
        """
        Decorator for exponential backoff retry logic
        
        Args:
            func: Function to wrap
            max_retries: Maximum number of retry attempts
            base_delay: Base delay in seconds
            max_delay: Maximum delay in seconds
            exceptions: Tuple of exception types to catch
            
        Returns:
            Wrapped function
        """
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                    
                except exceptions as e:
                    last_exception = e
                    
                    if attempt == max_retries:
                        logger.error(f"Max retries ({max_retries}) reached for {func.__name__}")
                        raise
                    
                    # Calculate delay with exponential backoff
                    delay = min(base_delay * (2 ** attempt), max_delay)
                    
                    logger.warning(
                        f"Attempt {attempt + 1}/{max_retries} failed for {func.__name__}: {e}. "
                        f"Retrying in {delay:.1f}s..."
                    )
                    
                    time.sleep(delay)
            
            raise last_exception
        
        return wrapper


class ErrorRecovery:
    """Error recovery strategies"""
    
    @staticmethod
    def with_fallback(primary_func: Callable, fallback_func: Callable) -> Callable:
        """
        Try primary function, fall back to alternative on failure
        
        Args:
            primary_func: Primary function to try
            fallback_func: Fallback function if primary fails
            
        Returns:
            Wrapped function
        """
        @functools.wraps(primary_func)
        def wrapper(*args, **kwargs):
            try:
                return primary_func(*args, **kwargs)
            except Exception as e:
                logger.warning(f"Primary function failed: {e}. Using fallback.")
                return fallback_func(*args, **kwargs)
        
        return wrapper
    
    @staticmethod
    def graceful_degradation(func: Callable, default_value: Any = None) -> Callable:
        """
        Return default value on error instead of raising
        
        Args:
            func: Function to wrap
            default_value: Value to return on error
            
        Returns:
            Wrapped function
        """
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(f"Function {func.__name__} failed: {e}. Returning default value.")
                return default_value
        
        return wrapper


# Example usage decorators
def retry_on_api_error(max_retries: int = 3):
    """Decorator for retrying API calls"""
    def decorator(func):
        return RetryHandler.exponential_backoff(
            func,
            max_retries=max_retries,
            base_delay=1.0,
            exceptions=(ConnectionError, TimeoutError)
        )
    return decorator


def with_timeout(timeout_seconds: float):
    """Decorator for adding timeout to function"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            import signal
            
            def timeout_handler(signum, frame):
                raise TimeoutError(f"Function {func.__name__} timed out after {timeout_seconds}s")
            
            # Set alarm
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(int(timeout_seconds))
            
            try:
                result = func(*args, **kwargs)
            finally:
                signal.alarm(0)  # Cancel alarm
            
            return result
        
        return wrapper
    return decorator