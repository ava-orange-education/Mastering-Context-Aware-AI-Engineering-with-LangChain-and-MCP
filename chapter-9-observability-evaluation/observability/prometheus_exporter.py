"""
Prometheus metrics exporter for monitoring.
"""

from typing import Dict, Any, Optional
from prometheus_client import Counter, Histogram, Gauge, Summary, CollectorRegistry, generate_latest
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PrometheusExporter:
    """Export metrics to Prometheus"""
    
    def __init__(self, registry: Optional[CollectorRegistry] = None):
        """
        Initialize Prometheus exporter
        
        Args:
            registry: Optional custom registry
        """
        self.registry = registry or CollectorRegistry()
        
        # Agent execution metrics
        self.agent_requests_total = Counter(
            'agent_requests_total',
            'Total number of agent requests',
            ['agent_name', 'status'],
            registry=self.registry
        )
        
        self.agent_request_duration = Histogram(
            'agent_request_duration_seconds',
            'Agent request duration in seconds',
            ['agent_name'],
            registry=self.registry,
            buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0)
        )
        
        self.agent_errors_total = Counter(
            'agent_errors_total',
            'Total number of agent errors',
            ['agent_name', 'error_type'],
            registry=self.registry
        )
        
        # LLM metrics
        self.llm_calls_total = Counter(
            'llm_calls_total',
            'Total number of LLM API calls',
            ['model', 'status'],
            registry=self.registry
        )
        
        self.llm_tokens_total = Counter(
            'llm_tokens_total',
            'Total number of tokens used',
            ['model', 'type'],  # type: input/output
            registry=self.registry
        )
        
        self.llm_latency = Histogram(
            'llm_latency_seconds',
            'LLM API call latency in seconds',
            ['model'],
            registry=self.registry,
            buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0)
        )
        
        # Retrieval metrics
        self.retrieval_requests_total = Counter(
            'retrieval_requests_total',
            'Total number of retrieval requests',
            ['status'],
            registry=self.registry
        )
        
        self.retrieval_documents_count = Histogram(
            'retrieval_documents_count',
            'Number of documents retrieved',
            registry=self.registry,
            buckets=(1, 3, 5, 10, 20, 50)
        )
        
        self.retrieval_latency = Histogram(
            'retrieval_latency_seconds',
            'Retrieval latency in seconds',
            registry=self.registry,
            buckets=(0.01, 0.05, 0.1, 0.5, 1.0, 2.0)
        )
        
        # Evaluation metrics
        self.evaluation_score = Gauge(
            'evaluation_score',
            'Latest evaluation score',
            ['metric_name'],
            registry=self.registry
        )
        
        self.evaluation_runs_total = Counter(
            'evaluation_runs_total',
            'Total number of evaluation runs',
            registry=self.registry
        )
        
        # System metrics
        self.active_agents = Gauge(
            'active_agents',
            'Number of currently active agents',
            registry=self.registry
        )
        
        logger.info("Prometheus exporter initialized")
    
    def record_agent_request(self, agent_name: str, duration: float, 
                            status: str = 'success'):
        """
        Record agent request
        
        Args:
            agent_name: Agent name
            duration: Request duration in seconds
            status: Request status (success/failure)
        """
        self.agent_requests_total.labels(agent_name=agent_name, status=status).inc()
        self.agent_request_duration.labels(agent_name=agent_name).observe(duration)
        
        logger.debug(f"Recorded agent request: {agent_name} ({status}) - {duration:.3f}s")
    
    def record_agent_error(self, agent_name: str, error_type: str):
        """
        Record agent error
        
        Args:
            agent_name: Agent name
            error_type: Error type/category
        """
        self.agent_errors_total.labels(agent_name=agent_name, error_type=error_type).inc()
        logger.debug(f"Recorded agent error: {agent_name} - {error_type}")
    
    def record_llm_call(self, model: str, duration: float, 
                       input_tokens: int, output_tokens: int,
                       status: str = 'success'):
        """
        Record LLM API call
        
        Args:
            model: Model name
            duration: Call duration in seconds
            input_tokens: Input tokens used
            output_tokens: Output tokens used
            status: Call status
        """
        self.llm_calls_total.labels(model=model, status=status).inc()
        self.llm_latency.labels(model=model).observe(duration)
        self.llm_tokens_total.labels(model=model, type='input').inc(input_tokens)
        self.llm_tokens_total.labels(model=model, type='output').inc(output_tokens)
        
        logger.debug(f"Recorded LLM call: {model} - {input_tokens + output_tokens} tokens")
    
    def record_retrieval(self, duration: float, document_count: int,
                        status: str = 'success'):
        """
        Record retrieval operation
        
        Args:
            duration: Retrieval duration in seconds
            document_count: Number of documents retrieved
            status: Retrieval status
        """
        self.retrieval_requests_total.labels(status=status).inc()
        self.retrieval_latency.observe(duration)
        self.retrieval_documents_count.observe(document_count)
        
        logger.debug(f"Recorded retrieval: {document_count} docs in {duration:.3f}s")
    
    def record_evaluation_score(self, metric_name: str, score: float):
        """
        Record evaluation score
        
        Args:
            metric_name: Metric name (relevance, coherence, etc.)
            score: Score value
        """
        self.evaluation_score.labels(metric_name=metric_name).set(score)
        logger.debug(f"Recorded evaluation score: {metric_name} = {score:.3f}")
    
    def record_evaluation_run(self):
        """Record that an evaluation run occurred"""
        self.evaluation_runs_total.inc()
    
    def set_active_agents(self, count: int):
        """
        Set number of active agents
        
        Args:
            count: Number of active agents
        """
        self.active_agents.set(count)
    
    def get_metrics(self) -> bytes:
        """
        Get metrics in Prometheus format
        
        Returns:
            Metrics as bytes
        """
        return generate_latest(self.registry)
    
    def get_metrics_text(self) -> str:
        """
        Get metrics as text
        
        Returns:
            Metrics as string
        """
        return self.get_metrics().decode('utf-8')


class PrometheusServer:
    """HTTP server for Prometheus metrics"""
    
    def __init__(self, exporter: PrometheusExporter, port: int = 8000):
        """
        Initialize Prometheus server
        
        Args:
            exporter: PrometheusExporter instance
            port: Server port
        """
        self.exporter = exporter
        self.port = port
        self.server = None
    
    def start(self):
        """Start metrics server"""
        from prometheus_client import start_http_server
        
        try:
            start_http_server(self.port, registry=self.exporter.registry)
            logger.info(f"Prometheus metrics server started on port {self.port}")
            logger.info(f"Metrics available at http://localhost:{self.port}/metrics")
        except Exception as e:
            logger.error(f"Failed to start metrics server: {e}")
    
    def stop(self):
        """Stop metrics server"""
        # Prometheus client doesn't provide a stop method
        # Server runs in background thread
        logger.info("Prometheus server stop requested (runs in background)")