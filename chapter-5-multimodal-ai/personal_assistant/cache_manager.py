"""
Cache manager for personal assistant.
"""

from typing import Dict, Any, Optional
import hashlib
import json
import pickle
from pathlib import Path
import time
import logging

logger = logging.getLogger(__name__)


class MultiModalCache:
    """Cache for multi-modal assistant responses"""
    
    def __init__(self, cache_dir: str = "./cache", ttl_seconds: int = 3600):
        """
        Initialize cache
        
        Args:
            cache_dir: Directory for cache files
            ttl_seconds: Time-to-live for cache entries
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl = ttl_seconds
        
        self.stats = {
            'hits': 0,
            'misses': 0,
            'sets': 0
        }
    
    def _generate_key(self, request: Dict[str, Any]) -> str:
        """
        Generate cache key from request
        
        Args:
            request: Request dictionary
            
        Returns:
            Cache key
        """
        # For file-based requests, hash the file content
        key_data = {}
        
        for field in ['task', 'query', 'prompt', 'question']:
            if field in request:
                key_data[field] = request[field]
        
        # Hash file contents if present
        for file_field in ['image_path', 'audio_path', 'document_path']:
            if file_field in request:
                file_path = request[file_field]
                try:
                    with open(file_path, 'rb') as f:
                        file_hash = hashlib.md5(f.read()).hexdigest()
                    key_data[file_field] = file_hash
                except:
                    key_data[file_field] = file_path
        
        # Generate key
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_str.encode()).hexdigest()
    
    def get(self, request: Dict[str, Any]) -> Optional[Any]:
        """
        Get cached response
        
        Args:
            request: Request dictionary
            
        Returns:
            Cached response or None
        """
        key = self._generate_key(request)
        cache_file = self.cache_dir / f"{key}.pkl"
        
        if not cache_file.exists():
            self.stats['misses'] += 1
            return None
        
        try:
            with open(cache_file, 'rb') as f:
                cache_data = pickle.load(f)
            
            # Check TTL
            age = time.time() - cache_data['timestamp']
            if age > self.ttl:
                cache_file.unlink()
                self.stats['misses'] += 1
                return None
            
            self.stats['hits'] += 1
            logger.info(f"Cache hit for key: {key[:16]}...")
            return cache_data['response']
            
        except Exception as e:
            logger.error(f"Cache read error: {e}")
            self.stats['misses'] += 1
            return None
    
    def set(self, request: Dict[str, Any], response: Any):
        """
        Cache response
        
        Args:
            request: Request dictionary
            response: Response to cache
        """
        key = self._generate_key(request)
        cache_file = self.cache_dir / f"{key}.pkl"
        
        cache_data = {
            'request': request,
            'response': response,
            'timestamp': time.time()
        }
        
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            
            self.stats['sets'] += 1
            logger.info(f"Cached response for key: {key[:16]}...")
            
        except Exception as e:
            logger.error(f"Cache write error: {e}")
    
    def clear(self):
        """Clear all cache"""
        for cache_file in self.cache_dir.glob("*.pkl"):
            cache_file.unlink()
        
        logger.info("Cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_requests = self.stats['hits'] + self.stats['misses']
        hit_rate = self.stats['hits'] / total_requests if total_requests > 0 else 0
        
        return {
            **self.stats,
            'hit_rate': hit_rate,
            'cache_size': len(list(self.cache_dir.glob("*.pkl")))
        }