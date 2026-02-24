"""
Base classes for data ingestion pipelines.
"""

from typing import Dict, Any, List, Optional, Iterator
from dataclasses import dataclass, field
from datetime import datetime
from abc import ABC, abstractmethod
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DataSource:
    """Configuration for a data source"""
    source_id: str
    source_type: str  # 'database', 'api', 'file', 'stream'
    connection_params: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class IngestionRecord:
    """Single record from data ingestion"""
    record_id: str
    source_id: str
    content: Any
    metadata: Dict[str, Any]
    timestamp: datetime
    checksum: Optional[str] = None


class DataConnector(ABC):
    """Abstract base class for data source connectors"""
    
    def __init__(self, source: DataSource):
        self.source = source
        self.last_checkpoint = None
    
    @abstractmethod
    def connect(self) -> bool:
        """Establish connection to data source"""
        pass
    
    @abstractmethod
    def extract(self, checkpoint: Optional[Any] = None) -> Iterator[IngestionRecord]:
        """Extract data from source with optional checkpoint"""
        pass
    
    @abstractmethod
    def get_checkpoint(self) -> Any:
        """Get current checkpoint for incremental extraction"""
        pass
    
    @abstractmethod
    def close(self):
        """Close connection to data source"""
        pass


class IngestionPipeline:
    """Base ingestion pipeline orchestrator"""
    
    def __init__(self, pipeline_id: str):
        self.pipeline_id = pipeline_id
        self.connectors: Dict[str, DataConnector] = {}
        self.stats = {
            'records_processed': 0,
            'records_failed': 0,
            'sources_processed': 0
        }
    
    def register_connector(self, connector: DataConnector):
        """Register a data connector"""
        self.connectors[connector.source.source_id] = connector
        logger.info(f"Registered connector: {connector.source.source_id}")
    
    def run(self, checkpoint_store: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Execute ingestion pipeline
        
        Args:
            checkpoint_store: Dictionary storing checkpoints by source_id
            
        Returns:
            Pipeline execution statistics
        """
        checkpoint_store = checkpoint_store or {}
        
        for source_id, connector in self.connectors.items():
            try:
                logger.info(f"Processing source: {source_id}")
                
                # Connect to source
                if not connector.connect():
                    logger.error(f"Failed to connect to {source_id}")
                    continue
                
                # Get last checkpoint
                checkpoint = checkpoint_store.get(source_id)
                
                # Extract records
                for record in connector.extract(checkpoint):
                    try:
                        self._process_record(record)
                        self.stats['records_processed'] += 1
                    except Exception as e:
                        logger.error(f"Failed to process record {record.record_id}: {e}")
                        self.stats['records_failed'] += 1
                
                # Update checkpoint
                checkpoint_store[source_id] = connector.get_checkpoint()
                self.stats['sources_processed'] += 1
                
                # Close connection
                connector.close()
                
            except Exception as e:
                logger.error(f"Error processing source {source_id}: {e}")
        
        return self.stats
    
    def _process_record(self, record: IngestionRecord):
        """
        Process individual record (override in subclass)
        
        Args:
            record: Record to process
        """
        pass