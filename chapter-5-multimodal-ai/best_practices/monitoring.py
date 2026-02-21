"""
Performance monitoring and metrics collection.
"""

from typing import Dict, Any, List
import time
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class MetricsCollector:
    """Collect system metrics"""
    
    def __init__(self):
        self.metrics = {
            'requests': [],
            'errors': [],
            'latencies': []
        }
    
    def record_request(self, 
                      task: str,
                      latency: float,
                      success: bool,
                      error: Optional[str] = None):
        """
        Record a request
        
        Args:
            task: Task type
            latency: Request latency in seconds
            success: Whether request succeeded
            error: Error message if failed
        """
        self.metrics['requests'].append({
            'timestamp': datetime.now(),
            'task': task,
            'latency': latency,
            'success': success
        })
        
        self.metrics['latencies'].append(latency)
        
        if not success:
            self.metrics['errors'].append({
                'timestamp': datetime.now(),
                'task': task,
                'error': error
            })
    
    def get_metrics(self, time_window_minutes: int = 60) -> Dict[str, Any]:
        """
        Get metrics for time window
        
        Args:
            time_window_minutes: Time window in minutes
            
        Returns:
            Metrics summary
        """
        import numpy as np
        from datetime import timedelta
        
        cutoff = datetime.now() - timedelta(minutes=time_window_minutes)
        
        recent_requests = [
            r for r in self.metrics['requests']
            if r['timestamp'] > cutoff
        ]
        
        if not recent_requests:
            return {'message': 'No recent requests'}
        
        recent_latencies = [r['latency'] for r in recent_requests]
        successful = [r for r in recent_requests if r['success']]
        
        return {
            'time_window_minutes': time_window_minutes,
            'total_requests': len(recent_requests),
            'successful_requests': len(successful),
            'failed_requests': len(recent_requests) - len(successful),
            'success_rate': len(successful) / len(recent_requests),
            'latency': {
                'mean': np.mean(recent_latencies),
                'median': np.median(recent_latencies),
                'p95': np.percentile(recent_latencies, 95),
                'p99': np.percentile(recent_latencies, 99),
                'min': np.min(recent_latencies),
                'max': np.max(recent_latencies)
            }
        }


class PerformanceMonitor:
    """Monitor assistant performance"""
    
    def __init__(self, assistant):
        """
        Initialize monitor
        
        Args:
            assistant: Assistant instance to monitor
        """
        self.assistant = assistant
        self.metrics = MetricsCollector()
    
    def monitored_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process request with monitoring
        
        Args:
            request: Request dictionary
            
        Returns:
            Result with monitoring metadata
        """
        start_time = time.time()
        task = request.get('task', 'unknown')
        
        try:
            result = self.assistant.process_request(request)
            latency = time.time() - start_time
            
            success = result.get('success', False)
            error = result.get('error')
            
            self.metrics.record_request(task, latency, success, error)
            
            result['monitoring'] = {
                'latency': latency,
                'timestamp': datetime.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            latency = time.time() - start_time
            self.metrics.record_request(task, latency, False, str(e))
            raise
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report"""
        metrics = self.metrics.get_metrics()
        
        # Add assistant-specific stats
        assistant_stats = self.assistant.get_stats()
        
        return {
            'metrics': metrics,
            'assistant_stats': assistant_stats,
            'generated_at': datetime.now().isoformat()
        }
    
    def check_health(self) -> Dict[str, Any]:
        """Check system health"""
        metrics = self.metrics.get_metrics(time_window_minutes=5)
        
        if not metrics or metrics.get('total_requests', 0) == 0:
            return {
                'status': 'unknown',
                'message': 'No recent activity'
            }
        
        success_rate = metrics.get('success_rate', 0)
        p95_latency = metrics.get('latency', {}).get('p95', 0)
        
        # Determine health status
        if success_rate >= 0.99 and p95_latency < 5.0:
            status = 'healthy'
        elif success_rate >= 0.95 and p95_latency < 10.0:
            status = 'degraded'
        else:
            status = 'unhealthy'
        
        return {
            'status': status,
            'success_rate': success_rate,
            'p95_latency': p95_latency,
            'metrics': metrics
        }