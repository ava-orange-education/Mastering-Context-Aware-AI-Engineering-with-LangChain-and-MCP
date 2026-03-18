"""
Custom metrics collection
"""

from prometheus_client import Counter, Histogram, Gauge
import time
from functools import wraps
import logging

logger = logging.getLogger(__name__)

# Define metrics
query_counter = Counter(
    'rag_queries_total',
    'Total number of queries processed',
    ['agent_type', 'status']
)

query_duration = Histogram(
    'rag_query_duration_seconds',
    'Query processing duration',
    ['agent_type']
)

active_requests = Gauge(
    'rag_active_requests',
    'Number of requests currently being processed'
)

agent_errors = Counter(
    'rag_agent_errors_total',
    'Total number of agent errors',
    ['agent_name', 'error_type']
)


class MetricsCollector:
    """Collect custom application metrics"""
    
    @staticmethod
    def record_query(agent_type: str, status: str):
        """Record a query execution"""
        query_counter.labels(agent_type=agent_type, status=status).inc()
    
    @staticmethod
    def record_query_duration(agent_type: str, duration: float):
        """Record query duration"""
        query_duration.labels(agent_type=agent_type).observe(duration)
    
    @staticmethod
    def record_error(agent_name: str, error_type: str):
        """Record an agent error"""
        agent_errors.labels(agent_name=agent_name, error_type=error_type).inc()
    
    @staticmethod
    def track_request(func):
        """Decorator to track request metrics"""
        @wraps(func)
        async def wrapper(*args, **kwargs):
            active_requests.inc()
            start_time = time.time()
            
            try:
                result = await func(*args, **kwargs)
                duration = time.time() - start_time
                
                # Extract agent_type from result if available
                agent_type = getattr(result, 'agent_type', 'unknown')
                MetricsCollector.record_query_duration(agent_type, duration)
                MetricsCollector.record_query(agent_type, 'success')
                
                return result
            
            except Exception as e:
                duration = time.time() - start_time
                logger.error(f"Request failed after {duration:.2f}s: {e}")
                
                MetricsCollector.record_query('unknown', 'error')
                raise
            
            finally:
                active_requests.dec()
        
        return wrapper