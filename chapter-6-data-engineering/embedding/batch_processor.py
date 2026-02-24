"""
Batch processing for efficient embedding generation.
"""

from typing import List, Dict, Any, Optional, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class BatchJob:
    """Represents a batch processing job"""
    job_id: str
    texts: List[str]
    metadata: List[Dict[str, Any]]
    status: str = "pending"  # pending, processing, completed, failed
    results: Optional[List[Any]] = None
    error: Optional[str] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None


class BatchEmbeddingProcessor:
    """Process embeddings in batches with parallel execution"""
    
    def __init__(self,
                 embedding_manager,
                 batch_size: int = 100,
                 max_workers: int = 4):
        """
        Initialize batch processor
        
        Args:
            embedding_manager: EmbeddingManager instance
            batch_size: Number of texts per batch
            max_workers: Number of parallel workers
        """
        self.embedding_manager = embedding_manager
        self.batch_size = batch_size
        self.max_workers = max_workers
        
        self.jobs: Dict[str, BatchJob] = {}
        self.job_counter = 0
    
    def submit_job(self,
                   texts: List[str],
                   metadata: Optional[List[Dict[str, Any]]] = None) -> str:
        """
        Submit a batch job
        
        Args:
            texts: List of texts to embed
            metadata: Optional metadata for each text
            
        Returns:
            Job ID
        """
        job_id = f"job_{self.job_counter}"
        self.job_counter += 1
        
        if metadata is None:
            metadata = [{} for _ in texts]
        
        job = BatchJob(
            job_id=job_id,
            texts=texts,
            metadata=metadata
        )
        
        self.jobs[job_id] = job
        
        logger.info(f"Submitted job {job_id} with {len(texts)} texts")
        
        return job_id
    
    def process_job(self, job_id: str) -> BatchJob:
        """
        Process a submitted job
        
        Args:
            job_id: Job ID to process
            
        Returns:
            Completed BatchJob
        """
        if job_id not in self.jobs:
            raise ValueError(f"Job {job_id} not found")
        
        job = self.jobs[job_id]
        
        if job.status != "pending":
            logger.warning(f"Job {job_id} already processed")
            return job
        
        job.status = "processing"
        job.start_time = time.time()
        
        try:
            # Split into batches
            batches = self._create_batches(job.texts, job.metadata)
            
            # Process batches in parallel
            all_results = []
            
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_batch = {
                    executor.submit(self._process_batch, batch_texts, batch_meta): i
                    for i, (batch_texts, batch_meta) in enumerate(batches)
                }
                
                for future in as_completed(future_to_batch):
                    batch_idx = future_to_batch[future]
                    
                    try:
                        results = future.result()
                        all_results.extend(results)
                        logger.info(f"Completed batch {batch_idx + 1}/{len(batches)}")
                    except Exception as e:
                        logger.error(f"Batch {batch_idx} failed: {e}")
                        raise
            
            job.results = all_results
            job.status = "completed"
            job.end_time = time.time()
            
            duration = job.end_time - job.start_time
            logger.info(f"Job {job_id} completed in {duration:.2f}s")
        
        except Exception as e:
            job.status = "failed"
            job.error = str(e)
            job.end_time = time.time()
            logger.error(f"Job {job_id} failed: {e}")
        
        return job
    
    def _create_batches(self,
                       texts: List[str],
                       metadata: List[Dict[str, Any]]) -> List[tuple]:
        """Create batches from texts and metadata"""
        batches = []
        
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            batch_meta = metadata[i:i + self.batch_size]
            batches.append((batch_texts, batch_meta))
        
        return batches
    
    def _process_batch(self,
                      texts: List[str],
                      metadata: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process a single batch"""
        # Generate embeddings
        embedding_results = self.embedding_manager.embed_batch(texts)
        
        # Combine with metadata
        results = []
        for emb_result, meta in zip(embedding_results, metadata):
            result = {
                'text': emb_result.text,
                'embedding': emb_result.embedding.tolist(),
                'metadata': {**meta, **emb_result.metadata}
            }
            results.append(result)
        
        return results
    
    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """Get status of a job"""
        if job_id not in self.jobs:
            return {'error': f'Job {job_id} not found'}
        
        job = self.jobs[job_id]
        
        status = {
            'job_id': job.job_id,
            'status': job.status,
            'num_texts': len(job.texts),
            'num_results': len(job.results) if job.results else 0
        }
        
        if job.start_time:
            status['start_time'] = job.start_time
        
        if job.end_time:
            status['end_time'] = job.end_time
            status['duration'] = job.end_time - job.start_time
        
        if job.error:
            status['error'] = job.error
        
        return status
    
    def process_all_pending(self) -> List[str]:
        """Process all pending jobs"""
        pending_jobs = [
            job_id for job_id, job in self.jobs.items()
            if job.status == "pending"
        ]
        
        completed_jobs = []
        
        for job_id in pending_jobs:
            job = self.process_job(job_id)
            if job.status == "completed":
                completed_jobs.append(job_id)
        
        return completed_jobs


class StreamingEmbeddingProcessor:
    """Process embeddings in streaming fashion for large datasets"""
    
    def __init__(self,
                 embedding_manager,
                 batch_size: int = 100,
                 callback: Optional[Callable] = None):
        """
        Initialize streaming processor
        
        Args:
            embedding_manager: EmbeddingManager instance
            batch_size: Number of texts per batch
            callback: Optional callback function(batch_results)
        """
        self.embedding_manager = embedding_manager
        self.batch_size = batch_size
        self.callback = callback
        
        self.processed_count = 0
        self.failed_count = 0
    
    def process_stream(self, text_iterator):
        """
        Process texts from an iterator
        
        Args:
            text_iterator: Iterator yielding (text, metadata) tuples
            
        Yields:
            Embedding results
        """
        batch_texts = []
        batch_metadata = []
        
        for text, metadata in text_iterator:
            batch_texts.append(text)
            batch_metadata.append(metadata)
            
            if len(batch_texts) >= self.batch_size:
                # Process batch
                try:
                    results = self._process_batch(batch_texts, batch_metadata)
                    
                    # Call callback if provided
                    if self.callback:
                        self.callback(results)
                    
                    # Yield results
                    for result in results:
                        yield result
                    
                    self.processed_count += len(batch_texts)
                
                except Exception as e:
                    logger.error(f"Batch processing failed: {e}")
                    self.failed_count += len(batch_texts)
                
                # Reset batch
                batch_texts = []
                batch_metadata = []
        
        # Process remaining texts
        if batch_texts:
            try:
                results = self._process_batch(batch_texts, batch_metadata)
                
                if self.callback:
                    self.callback(results)
                
                for result in results:
                    yield result
                
                self.processed_count += len(batch_texts)
            
            except Exception as e:
                logger.error(f"Final batch processing failed: {e}")
                self.failed_count += len(batch_texts)
    
    def _process_batch(self, texts: List[str], metadata: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process a single batch"""
        embedding_results = self.embedding_manager.embed_batch(texts)
        
        results = []
        for emb_result, meta in zip(embedding_results, metadata):
            result = {
                'text': emb_result.text,
                'embedding': emb_result.embedding,
                'metadata': {**meta, **emb_result.metadata}
            }
            results.append(result)
        
        return results
    
    def get_stats(self) -> Dict[str, int]:
        """Get processing statistics"""
        return {
            'processed': self.processed_count,
            'failed': self.failed_count,
            'total': self.processed_count + self.failed_count
        }