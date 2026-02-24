"""
Pipeline health monitoring and metrics collection.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import logging
import time

logger = logging.getLogger(__name__)


class PipelineMetrics:
    """Track pipeline execution metrics"""
    
    def __init__(self):
        self.metrics = {
            'executions': [],
            'latencies': [],
            'throughput': [],
            'error_rates': []
        }
    
    def record_execution(self, pipeline_id: str, status: str, 
                        duration: float, records_processed: int, errors: int):
        """Record pipeline execution"""
        self.metrics['executions'].append({
            'pipeline_id': pipeline_id,
            'timestamp': datetime.now().isoformat(),
            'status': status,
            'duration': duration,
            'records_processed': records_processed,
            'errors': errors
        })
        
        self.metrics['latencies'].append(duration)
        
        if records_processed > 0:
            throughput = records_processed / duration
            self.metrics['throughput'].append(throughput)
        
        error_rate = errors / records_processed if records_processed > 0 else 0
        self.metrics['error_rates'].append(error_rate)
    
    def get_summary(self, time_window_hours: int = 24) -> Dict[str, Any]:
        """Get metrics summary"""
        cutoff = datetime.now() - timedelta(hours=time_window_hours)
        
        recent_executions = [
            e for e in self.metrics['executions']
            if datetime.fromisoformat(e['timestamp']) > cutoff
        ]
        
        if not recent_executions:
            return {'message': 'No recent executions'}
        
        # Calculate statistics
        total_executions = len(recent_executions)
        successful = sum(1 for e in recent_executions if e['status'] == 'success')
        failed = total_executions - successful
        
        total_records = sum(e['records_processed'] for e in recent_executions)
        total_errors = sum(e['errors'] for e in recent_executions)
        
        import numpy as np
        latencies = [e['duration'] for e in recent_executions]
        
        return {
            'time_window_hours': time_window_hours,
            'total_executions': total_executions,
            'successful_executions': successful,
            'failed_executions': failed,
            'success_rate': successful / total_executions,
            'total_records_processed': total_records,
            'total_errors': total_errors,
            'overall_error_rate': total_errors / total_records if total_records > 0 else 0,
            'latency': {
                'mean': np.mean(latencies),
                'median': np.median(latencies),
                'p95': np.percentile(latencies, 95),
                'p99': np.percentile(latencies, 99),
                'min': np.min(latencies),
                'max': np.max(latencies)
            },
            'avg_throughput': np.mean(self.metrics['throughput']) if self.metrics['throughput'] else 0
        }


class PipelineMonitor:
    """Monitor pipeline health and performance"""
    
    def __init__(self):
        self.metrics = PipelineMetrics()
        self.alerts = []
        self.alert_thresholds = {
            'error_rate': 0.05,  # 5%
            'success_rate': 0.95,  # 95%
            'max_latency': 3600  # 1 hour
        }
    
    def set_alert_threshold(self, metric: str, threshold: float):
        """Set alert threshold"""
        self.alert_thresholds[metric] = threshold
        logger.info(f"Set alert threshold: {metric} = {threshold}")
    
    def monitor_execution(self, pipeline_func, pipeline_id: str, *args, **kwargs) -> Dict[str, Any]:
        """
        Monitor pipeline execution
        
        Args:
            pipeline_func: Pipeline function to execute
            pipeline_id: Pipeline identifier
            *args, **kwargs: Arguments to pass to pipeline function
            
        Returns:
            Execution result with monitoring data
        """
        start_time = time.time()
        
        try:
            result = pipeline_func(*args, **kwargs)
            
            duration = time.time() - start_time
            status = result.get('status', 'success')
            records_processed = result.get('records_processed', 0)
            errors = len(result.get('errors', []))
            
            # Record metrics
            self.metrics.record_execution(
                pipeline_id=pipeline_id,
                status=status,
                duration=duration,
                records_processed=records_processed,
                errors=errors
            )
            
            # Check for alerts
            self._check_alerts(pipeline_id, status, duration, records_processed, errors)
            
            # Add monitoring metadata to result
            result['monitoring'] = {
                'pipeline_id': pipeline_id,
                'duration': duration,
                'timestamp': datetime.now().isoformat()
            }
            
            return result
        
        except Exception as e:
            duration = time.time() - start_time
            
            logger.error(f"Pipeline execution failed: {e}")
            
            # Record failure
            self.metrics.record_execution(
                pipeline_id=pipeline_id,
                status='failed',
                duration=duration,
                records_processed=0,
                errors=1
            )
            
            # Create alert
            self._create_alert(
                'execution_failure',
                f"Pipeline {pipeline_id} failed: {str(e)}",
                'critical'
            )
            
            raise
    
    def _check_alerts(self, pipeline_id: str, status: str, 
                     duration: float, records_processed: int, errors: int):
        """Check if alerts should be triggered"""
        # Check error rate
        if records_processed > 0:
            error_rate = errors / records_processed
            
            if error_rate > self.alert_thresholds['error_rate']:
                self._create_alert(
                    'high_error_rate',
                    f"Pipeline {pipeline_id} error rate: {error_rate:.2%}",
                    'warning'
                )
        
        # Check latency
        if duration > self.alert_thresholds['max_latency']:
            self._create_alert(
                'high_latency',
                f"Pipeline {pipeline_id} took {duration:.0f}s",
                'warning'
            )
        
        # Check success status
        if status != 'success':
            self._create_alert(
                'execution_failure',
                f"Pipeline {pipeline_id} failed",
                'error'
            )
    
    def _create_alert(self, alert_type: str, message: str, severity: str):
        """Create alert"""
        alert = {
            'type': alert_type,
            'message': message,
            'severity': severity,
            'timestamp': datetime.now().isoformat()
        }
        
        self.alerts.append(alert)
        logger.warning(f"Alert: [{severity}] {message}")
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get overall pipeline health status"""
        summary = self.metrics.get_summary(time_window_hours=24)
        
        if 'message' in summary:
            return {
                'status': 'unknown',
                'message': summary['message']
            }
        
        # Determine health status
        success_rate = summary['success_rate']
        error_rate = summary['overall_error_rate']
        
        if success_rate >= 0.99 and error_rate < 0.01:
            status = 'healthy'
        elif success_rate >= 0.95 and error_rate < 0.05:
            status = 'degraded'
        else:
            status = 'unhealthy'
        
        # Count recent alerts
        recent_alerts = [
            a for a in self.alerts
            if datetime.fromisoformat(a['timestamp']) > datetime.now() - timedelta(hours=1)
        ]
        
        return {
            'status': status,
            'success_rate': success_rate,
            'error_rate': error_rate,
            'avg_latency': summary['latency']['mean'],
            'recent_alerts': len(recent_alerts),
            'alert_summary': {
                'critical': sum(1 for a in recent_alerts if a['severity'] == 'critical'),
                'error': sum(1 for a in recent_alerts if a['severity'] == 'error'),
                'warning': sum(1 for a in recent_alerts if a['severity'] == 'warning')
            },
            'metrics_summary': summary
        }
    
    def get_alerts(self, severity: Optional[str] = None, last_n: Optional[int] = 10) -> List[Dict[str, Any]]:
        """
        Get recent alerts
        
        Args:
            severity: Optional filter by severity
            last_n: Number of recent alerts to return
            
        Returns:
            List of alerts
        """
        alerts = self.alerts
        
        if severity:
            alerts = [a for a in alerts if a['severity'] == severity]
        
        # Sort by timestamp descending
        alerts = sorted(alerts, key=lambda x: x['timestamp'], reverse=True)
        
        if last_n:
            alerts = alerts[:last_n]
        
        return alerts