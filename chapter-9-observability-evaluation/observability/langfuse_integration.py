"""
Langfuse integration for observability and tracing.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LangfuseIntegration:
    """Integration with Langfuse for tracing and observability"""
    
    def __init__(self, public_key: Optional[str] = None, 
                 secret_key: Optional[str] = None,
                 host: Optional[str] = None):
        """
        Initialize Langfuse integration
        
        Args:
            public_key: Langfuse public key
            secret_key: Langfuse secret key
            host: Langfuse host URL
        """
        self.public_key = public_key
        self.secret_key = secret_key
        self.host = host or "https://cloud.langfuse.com"
        self.enabled = public_key is not None and secret_key is not None
        
        if self.enabled:
            try:
                from langfuse import Langfuse
                self.client = Langfuse(
                    public_key=public_key,
                    secret_key=secret_key,
                    host=host
                )
                logger.info("Langfuse integration initialized")
            except ImportError:
                logger.warning("Langfuse not installed. Install with: pip install langfuse")
                self.enabled = False
            except Exception as e:
                logger.error(f"Langfuse initialization failed: {e}")
                self.enabled = False
        else:
            logger.info("Langfuse integration disabled (no credentials provided)")
            self.client = None
    
    def trace_agent_run(self, agent_name: str, query: str, response: str,
                       metadata: Optional[Dict] = None) -> Optional[str]:
        """
        Trace an agent run
        
        Args:
            agent_name: Name of the agent
            query: Input query
            response: Agent response
            metadata: Additional metadata
            
        Returns:
            Trace ID if successful
        """
        if not self.enabled:
            return None
        
        try:
            trace = self.client.trace(
                name=f"{agent_name}_run",
                input=query,
                output=response,
                metadata=metadata or {}
            )
            
            logger.info(f"Trace created: {trace.id}")
            return trace.id
        
        except Exception as e:
            logger.error(f"Failed to create trace: {e}")
            return None
    
    def trace_generation(self, model: str, prompt: str, completion: str,
                        token_count: Optional[int] = None,
                        latency_ms: Optional[float] = None,
                        metadata: Optional[Dict] = None) -> Optional[str]:
        """
        Trace LLM generation
        
        Args:
            model: Model name
            prompt: Input prompt
            completion: Model completion
            token_count: Total tokens used
            latency_ms: Latency in milliseconds
            metadata: Additional metadata
            
        Returns:
            Generation ID if successful
        """
        if not self.enabled:
            return None
        
        try:
            generation = self.client.generation(
                name=f"{model}_generation",
                model=model,
                prompt=prompt,
                completion=completion,
                metadata=metadata or {}
            )
            
            if token_count:
                generation.usage = {"total_tokens": token_count}
            
            if latency_ms:
                generation.latency = latency_ms
            
            logger.info(f"Generation traced: {generation.id}")
            return generation.id
        
        except Exception as e:
            logger.error(f"Failed to trace generation: {e}")
            return None
    
    def trace_retrieval(self, query: str, documents: List[Dict],
                       metadata: Optional[Dict] = None) -> Optional[str]:
        """
        Trace document retrieval
        
        Args:
            query: Search query
            documents: Retrieved documents
            metadata: Additional metadata
            
        Returns:
            Span ID if successful
        """
        if not self.enabled:
            return None
        
        try:
            span = self.client.span(
                name="retrieval",
                input=query,
                output={"document_count": len(documents), "documents": documents},
                metadata=metadata or {}
            )
            
            logger.info(f"Retrieval traced: {span.id}")
            return span.id
        
        except Exception as e:
            logger.error(f"Failed to trace retrieval: {e}")
            return None
    
    def log_score(self, trace_id: str, name: str, value: float,
                 comment: Optional[str] = None):
        """
        Log evaluation score for a trace
        
        Args:
            trace_id: Trace ID
            name: Score name (e.g., 'relevance', 'accuracy')
            value: Score value
            comment: Optional comment
        """
        if not self.enabled:
            return
        
        try:
            self.client.score(
                trace_id=trace_id,
                name=name,
                value=value,
                comment=comment
            )
            
            logger.info(f"Score logged: {name}={value} for trace {trace_id}")
        
        except Exception as e:
            logger.error(f"Failed to log score: {e}")
    
    def create_dataset(self, name: str, description: Optional[str] = None) -> Optional[str]:
        """
        Create evaluation dataset
        
        Args:
            name: Dataset name
            description: Dataset description
            
        Returns:
            Dataset ID if successful
        """
        if not self.enabled:
            return None
        
        try:
            dataset = self.client.create_dataset(
                name=name,
                description=description
            )
            
            logger.info(f"Dataset created: {dataset.id}")
            return dataset.id
        
        except Exception as e:
            logger.error(f"Failed to create dataset: {e}")
            return None
    
    def add_dataset_item(self, dataset_id: str, input_data: Dict,
                        expected_output: Optional[Dict] = None,
                        metadata: Optional[Dict] = None):
        """
        Add item to dataset
        
        Args:
            dataset_id: Dataset ID
            input_data: Input data
            expected_output: Expected output
            metadata: Additional metadata
        """
        if not self.enabled:
            return
        
        try:
            self.client.create_dataset_item(
                dataset_id=dataset_id,
                input=input_data,
                expected_output=expected_output,
                metadata=metadata or {}
            )
            
            logger.info(f"Item added to dataset {dataset_id}")
        
        except Exception as e:
            logger.error(f"Failed to add dataset item: {e}")
    
    def flush(self):
        """Flush pending traces"""
        if self.enabled and self.client:
            try:
                self.client.flush()
                logger.info("Langfuse traces flushed")
            except Exception as e:
                logger.error(f"Failed to flush traces: {e}")