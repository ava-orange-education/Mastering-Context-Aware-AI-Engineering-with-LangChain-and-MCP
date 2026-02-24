"""
Database source connector for structured data ingestion.
"""

import psycopg2
from psycopg2 import sql
from typing import Iterator, Optional, Any, Dict, List
from datetime import datetime
from .pipeline_base import DataConnector, DataSource, IngestionRecord
import hashlib
import logging

logger = logging.getLogger(__name__)


class DatabaseConnector(DataConnector):
    """Connector for PostgreSQL databases"""
    
    def __init__(self, source: DataSource):
        super().__init__(source)
        self.connection = None
        self.cursor = None
        
        # Extract connection parameters
        self.db_config = {
            'host': source.connection_params.get('host', 'localhost'),
            'port': source.connection_params.get('port', 5432),
            'database': source.connection_params['database'],
            'user': source.connection_params['user'],
            'password': source.connection_params['password']
        }
        
        # Query configuration
        self.table = source.connection_params['table']
        self.columns = source.connection_params.get('columns', ['*'])
        self.timestamp_column = source.connection_params.get('timestamp_column', 'updated_at')
        self.id_column = source.connection_params.get('id_column', 'id')
    
    def connect(self) -> bool:
        """Establish database connection"""
        try:
            self.connection = psycopg2.connect(**self.db_config)
            self.cursor = self.connection.cursor()
            logger.info(f"Connected to database: {self.db_config['database']}")
            return True
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            return False
    
    def extract(self, checkpoint: Optional[Any] = None) -> Iterator[IngestionRecord]:
        """
        Extract records from database
        
        Args:
            checkpoint: Last processed timestamp for incremental extraction
            
        Yields:
            IngestionRecord objects
        """
        if not self.cursor:
            raise RuntimeError("Not connected to database")
        
        # Build query
        columns_str = ', '.join(self.columns) if self.columns != ['*'] else '*'
        
        query = sql.SQL("""
            SELECT {columns}, {timestamp_col}, {id_col}
            FROM {table}
            WHERE {timestamp_col} > %s
            ORDER BY {timestamp_col} ASC
        """).format(
            columns=sql.SQL(columns_str),
            timestamp_col=sql.Identifier(self.timestamp_column),
            id_col=sql.Identifier(self.id_column),
            table=sql.Identifier(self.table)
        )
        
        # Use checkpoint or very old date
        checkpoint_value = checkpoint or datetime(1970, 1, 1)
        
        try:
            self.cursor.execute(query, (checkpoint_value,))
            
            # Get column names
            col_names = [desc[0] for desc in self.cursor.description]
            
            # Fetch and yield records
            batch_size = 1000
            while True:
                rows = self.cursor.fetchmany(batch_size)
                if not rows:
                    break
                
                for row in rows:
                    record_dict = dict(zip(col_names, row))
                    
                    # Extract metadata
                    record_id = str(record_dict[self.id_column])
                    timestamp = record_dict[self.timestamp_column]
                    
                    # Create content (exclude metadata columns)
                    content = {
                        k: v for k, v in record_dict.items()
                        if k not in [self.id_column, self.timestamp_column]
                    }
                    
                    # Generate checksum
                    checksum = self._generate_checksum(content)
                    
                    # Update checkpoint
                    self.last_checkpoint = timestamp
                    
                    yield IngestionRecord(
                        record_id=record_id,
                        source_id=self.source.source_id,
                        content=content,
                        metadata={
                            'table': self.table,
                            'timestamp': timestamp.isoformat(),
                            'columns': list(content.keys())
                        },
                        timestamp=timestamp,
                        checksum=checksum
                    )
        
        except Exception as e:
            logger.error(f"Error extracting from database: {e}")
            raise
    
    def get_checkpoint(self) -> Any:
        """Get current checkpoint (last processed timestamp)"""
        return self.last_checkpoint
    
    def close(self):
        """Close database connection"""
        if self.cursor:
            self.cursor.close()
        if self.connection:
            self.connection.close()
        logger.info(f"Closed connection to {self.db_config['database']}")
    
    def _generate_checksum(self, content: Dict) -> str:
        """Generate MD5 checksum of record content"""
        content_str = str(sorted(content.items()))
        return hashlib.md5(content_str.encode()).hexdigest()


class BigQueryConnector(DataConnector):
    """Connector for Google BigQuery"""
    
    def __init__(self, source: DataSource):
        super().__init__(source)
        self.client = None
        
        self.project_id = source.connection_params['project_id']
        self.dataset_id = source.connection_params['dataset_id']
        self.table_id = source.connection_params['table_id']
        self.credentials_path = source.connection_params.get('credentials_path')
    
    def connect(self) -> bool:
        """Establish BigQuery connection"""
        try:
            from google.cloud import bigquery
            
            if self.credentials_path:
                self.client = bigquery.Client.from_service_account_json(
                    self.credentials_path
                )
            else:
                self.client = bigquery.Client(project=self.project_id)
            
            logger.info(f"Connected to BigQuery project: {self.project_id}")
            return True
        
        except Exception as e:
            logger.error(f"BigQuery connection failed: {e}")
            return False
    
    def extract(self, checkpoint: Optional[Any] = None) -> Iterator[IngestionRecord]:
        """Extract records from BigQuery"""
        if not self.client:
            raise RuntimeError("Not connected to BigQuery")
        
        # Build query
        table_ref = f"{self.project_id}.{self.dataset_id}.{self.table_id}"
        
        query = f"""
            SELECT *
            FROM `{table_ref}`
            WHERE _PARTITIONTIME > @checkpoint
            ORDER BY _PARTITIONTIME ASC
        """
        
        # Configure query parameters
        checkpoint_value = checkpoint or datetime(1970, 1, 1)
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("checkpoint", "TIMESTAMP", checkpoint_value)
            ]
        )
        
        try:
            query_job = self.client.query(query, job_config=job_config)
            
            for row in query_job:
                record_dict = dict(row.items())
                
                # Generate record ID
                record_id = f"{self.table_id}_{hash(str(record_dict))}"
                
                # Get partition time as checkpoint
                partition_time = record_dict.get('_PARTITIONTIME', datetime.now())
                self.last_checkpoint = partition_time
                
                yield IngestionRecord(
                    record_id=record_id,
                    source_id=self.source.source_id,
                    content=record_dict,
                    metadata={
                        'project': self.project_id,
                        'dataset': self.dataset_id,
                        'table': self.table_id
                    },
                    timestamp=partition_time,
                    checksum=None
                )
        
        except Exception as e:
            logger.error(f"Error extracting from BigQuery: {e}")
            raise
    
    def get_checkpoint(self) -> Any:
        """Get current checkpoint"""
        return self.last_checkpoint
    
    def close(self):
        """Close BigQuery connection"""
        self.client = None
        logger.info("Closed BigQuery connection")