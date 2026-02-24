"""
REST API source connector for external API ingestion.
"""

import requests
from typing import Iterator, Optional, Any, Dict
from datetime import datetime
import time
from .pipeline_base import DataConnector, DataSource, IngestionRecord
import logging

logger = logging.getLogger(__name__)


class APIConnector(DataConnector):
    """Connector for REST APIs"""
    
    def __init__(self, source: DataSource):
        super().__init__(source)
        
        self.base_url = source.connection_params['base_url'].rstrip('/')
        self.endpoint = source.connection_params['endpoint']
        
        # Authentication
        self.auth_type = source.connection_params.get('auth_type')  # 'bearer', 'api_key', 'basic'
        self.auth_token = source.connection_params.get('auth_token')
        self.api_key = source.connection_params.get('api_key')
        self.api_key_header = source.connection_params.get('api_key_header', 'X-API-Key')
        
        # Pagination
        self.pagination_type = source.connection_params.get('pagination', 'offset')  # 'offset', 'cursor', 'page'
        self.page_size = source.connection_params.get('page_size', 100)
        
        # Rate limiting
        self.rate_limit = source.connection_params.get('rate_limit', 10)  # requests per second
        self.last_request_time = 0
        
        self.session = None
    
    def connect(self) -> bool:
        """Initialize API session"""
        try:
            self.session = requests.Session()
            
            # Set up authentication
            if self.auth_type == 'bearer':
                self.session.headers.update({
                    'Authorization': f'Bearer {self.auth_token}'
                })
            elif self.auth_type == 'api_key':
                self.session.headers.update({
                    self.api_key_header: self.api_key
                })
            
            self.session.headers.update({
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            })
            
            logger.info(f"Initialized API session for: {self.base_url}")
            return True
        
        except Exception as e:
            logger.error(f"API session initialization failed: {e}")
            return False
    
    def extract(self, checkpoint: Optional[Any] = None) -> Iterator[IngestionRecord]:
        """
        Extract records from API with pagination
        
        Args:
            checkpoint: Last processed cursor/offset/page
            
        Yields:
            IngestionRecord objects
        """
        if not self.session:
            raise RuntimeError("API session not initialized")
        
        if self.pagination_type == 'offset':
            yield from self._extract_offset_pagination(checkpoint)
        elif self.pagination_type == 'cursor':
            yield from self._extract_cursor_pagination(checkpoint)
        elif self.pagination_type == 'page':
            yield from self._extract_page_pagination(checkpoint)
        else:
            yield from self._extract_no_pagination()
    
    def _extract_offset_pagination(self, checkpoint: Optional[int]) -> Iterator[IngestionRecord]:
        """Extract with offset-based pagination"""
        offset = checkpoint or 0
        
        while True:
            # Rate limiting
            self._rate_limit()
            
            # Make request
            url = f"{self.base_url}/{self.endpoint}"
            params = {
                'limit': self.page_size,
                'offset': offset
            }
            
            try:
                response = self.session.get(url, params=params, timeout=30)
                response.raise_for_status()
                data = response.json()
                
                # Extract records
                records = data.get('results', data.get('data', []))
                
                if not records:
                    break
                
                for record in records:
                    yield self._create_record(record)
                    offset += 1
                    self.last_checkpoint = offset
                
                # Check if more pages exist
                if len(records) < self.page_size:
                    break
            
            except Exception as e:
                logger.error(f"API request failed at offset {offset}: {e}")
                break
    
    def _extract_cursor_pagination(self, checkpoint: Optional[str]) -> Iterator[IngestionRecord]:
        """Extract with cursor-based pagination"""
        cursor = checkpoint
        
        while True:
            self._rate_limit()
            
            url = f"{self.base_url}/{self.endpoint}"
            params = {'limit': self.page_size}
            
            if cursor:
                params['cursor'] = cursor
            
            try:
                response = self.session.get(url, params=params, timeout=30)
                response.raise_for_status()
                data = response.json()
                
                records = data.get('results', data.get('data', []))
                
                if not records:
                    break
                
                for record in records:
                    yield self._create_record(record)
                
                # Get next cursor
                cursor = data.get('next_cursor')
                self.last_checkpoint = cursor
                
                if not cursor:
                    break
            
            except Exception as e:
                logger.error(f"API request failed: {e}")
                break
    
    def _extract_page_pagination(self, checkpoint: Optional[int]) -> Iterator[IngestionRecord]:
        """Extract with page-based pagination"""
        page = checkpoint or 1
        
        while True:
            self._rate_limit()
            
            url = f"{self.base_url}/{self.endpoint}"
            params = {
                'page': page,
                'per_page': self.page_size
            }
            
            try:
                response = self.session.get(url, params=params, timeout=30)
                response.raise_for_status()
                data = response.json()
                
                records = data.get('results', data.get('data', []))
                
                if not records:
                    break
                
                for record in records:
                    yield self._create_record(record)
                
                page += 1
                self.last_checkpoint = page
                
                # Check if more pages exist
                total_pages = data.get('total_pages')
                if total_pages and page > total_pages:
                    break
                
                if len(records) < self.page_size:
                    break
            
            except Exception as e:
                logger.error(f"API request failed at page {page}: {e}")
                break
    
    def _extract_no_pagination(self) -> Iterator[IngestionRecord]:
        """Extract without pagination (single request)"""
        self._rate_limit()
        
        url = f"{self.base_url}/{self.endpoint}"
        
        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            # Handle both list and dict responses
            if isinstance(data, list):
                records = data
            else:
                records = data.get('results', data.get('data', [data]))
            
            for record in records:
                yield self._create_record(record)
        
        except Exception as e:
            logger.error(f"API request failed: {e}")
    
    def _create_record(self, record: Dict) -> IngestionRecord:
        """Create IngestionRecord from API response"""
        record_id = str(record.get('id', hash(str(record))))
        
        return IngestionRecord(
            record_id=record_id,
            source_id=self.source.source_id,
            content=record,
            metadata={
                'api_endpoint': self.endpoint,
                'extracted_at': datetime.now().isoformat()
            },
            timestamp=datetime.now()
        )
    
    def _rate_limit(self):
        """Enforce rate limiting"""
        if self.rate_limit:
            elapsed = time.time() - self.last_request_time
            min_interval = 1.0 / self.rate_limit
            
            if elapsed < min_interval:
                time.sleep(min_interval - elapsed)
            
            self.last_request_time = time.time()
    
    def get_checkpoint(self) -> Any:
        """Get current checkpoint"""
        return self.last_checkpoint
    
    def close(self):
        """Close API session"""
        if self.session:
            self.session.close()
        logger.info(f"Closed API session for {self.base_url}")