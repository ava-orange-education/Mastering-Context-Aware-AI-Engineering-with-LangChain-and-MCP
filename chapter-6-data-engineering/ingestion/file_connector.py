"""
File system source connector for document ingestion.
"""

import os
from pathlib import Path
from typing import Iterator, Optional, Any, Dict, List
from datetime import datetime
from .pipeline_base import DataConnector, DataSource, IngestionRecord
import hashlib
import mimetypes
import logging

logger = logging.getLogger(__name__)


class FileConnector(DataConnector):
    """Connector for local file system"""
    
    def __init__(self, source: DataSource):
        super().__init__(source)
        
        self.root_path = Path(source.connection_params['root_path'])
        self.file_patterns = source.connection_params.get('patterns', ['*'])
        self.recursive = source.connection_params.get('recursive', True)
        self.supported_types = source.connection_params.get(
            'file_types', 
            ['.txt', '.md', '.pdf', '.docx', '.html', '.json', '.csv']
        )
        
        self.processed_files = set()
    
    def connect(self) -> bool:
        """Verify file system access"""
        try:
            if not self.root_path.exists():
                logger.error(f"Root path does not exist: {self.root_path}")
                return False
            
            if not self.root_path.is_dir():
                logger.error(f"Root path is not a directory: {self.root_path}")
                return False
            
            logger.info(f"Connected to file system: {self.root_path}")
            return True
        
        except Exception as e:
            logger.error(f"File system access failed: {e}")
            return False
    
    def extract(self, checkpoint: Optional[Any] = None) -> Iterator[IngestionRecord]:
        """
        Extract files from file system
        
        Args:
            checkpoint: Set of already processed file paths
            
        Yields:
            IngestionRecord objects
        """
        checkpoint = checkpoint or set()
        
        # Find matching files
        files = self._find_files()
        
        for file_path in files:
            # Skip if already processed
            file_str = str(file_path)
            if file_str in checkpoint:
                continue
            
            try:
                # Get file metadata
                stat = file_path.stat()
                
                # Read file content
                content = self._read_file(file_path)
                
                if content is None:
                    continue
                
                # Generate checksum
                checksum = self._generate_checksum(content)
                
                # Create record
                record = IngestionRecord(
                    record_id=checksum,
                    source_id=self.source.source_id,
                    content=content,
                    metadata={
                        'file_path': file_str,
                        'file_name': file_path.name,
                        'file_type': file_path.suffix,
                        'file_size': stat.st_size,
                        'created_at': datetime.fromtimestamp(stat.st_ctime).isoformat(),
                        'modified_at': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                        'mime_type': mimetypes.guess_type(file_path)[0]
                    },
                    timestamp=datetime.fromtimestamp(stat.st_mtime),
                    checksum=checksum
                )
                
                self.processed_files.add(file_str)
                yield record
            
            except Exception as e:
                logger.error(f"Error processing file {file_path}: {e}")
                continue
    
    def _find_files(self) -> List[Path]:
        """Find all matching files"""
        files = []
        
        if self.recursive:
            for pattern in self.file_patterns:
                files.extend(self.root_path.rglob(pattern))
        else:
            for pattern in self.file_patterns:
                files.extend(self.root_path.glob(pattern))
        
        # Filter by supported types
        files = [
            f for f in files
            if f.is_file() and f.suffix.lower() in self.supported_types
        ]
        
        return files
    
    def _read_file(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """Read file content based on type"""
        suffix = file_path.suffix.lower()
        
        try:
            if suffix in ['.txt', '.md', '.html']:
                return self._read_text_file(file_path)
            elif suffix == '.pdf':
                return self._read_pdf(file_path)
            elif suffix == '.docx':
                return self._read_docx(file_path)
            elif suffix == '.json':
                return self._read_json(file_path)
            elif suffix == '.csv':
                return self._read_csv(file_path)
            else:
                logger.warning(f"Unsupported file type: {suffix}")
                return None
        
        except Exception as e:
            logger.error(f"Error reading {file_path}: {e}")
            return None
    
    def _read_text_file(self, file_path: Path) -> Dict[str, Any]:
        """Read plain text file"""
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        return {
            'type': 'text',
            'content': content,
            'encoding': 'utf-8'
        }
    
    def _read_pdf(self, file_path: Path) -> Dict[str, Any]:
        """Read PDF file"""
        try:
            import PyPDF2
            
            text_content = []
            
            with open(file_path, 'rb') as f:
                pdf_reader = PyPDF2.PdfReader(f)
                
                for page_num, page in enumerate(pdf_reader.pages):
                    text = page.extract_text()
                    text_content.append({
                        'page': page_num + 1,
                        'text': text
                    })
            
            return {
                'type': 'pdf',
                'pages': text_content,
                'num_pages': len(text_content)
            }
        
        except ImportError:
            logger.error("PyPDF2 not installed. Install with: pip install PyPDF2")
            return None
    
    def _read_docx(self, file_path: Path) -> Dict[str, Any]:
        """Read DOCX file"""
        try:
            from docx import Document
            
            doc = Document(file_path)
            
            paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
            
            return {
                'type': 'docx',
                'paragraphs': paragraphs,
                'num_paragraphs': len(paragraphs)
            }
        
        except ImportError:
            logger.error("python-docx not installed. Install with: pip install python-docx")
            return None
    
    def _read_json(self, file_path: Path) -> Dict[str, Any]:
        """Read JSON file"""
        import json
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return {
            'type': 'json',
            'data': data
        }
    
    def _read_csv(self, file_path: Path) -> Dict[str, Any]:
        """Read CSV file"""
        import csv
        
        rows = []
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        
        return {
            'type': 'csv',
            'rows': rows,
            'num_rows': len(rows)
        }
    
    def _generate_checksum(self, content: Dict) -> str:
        """Generate MD5 checksum of content"""
        content_str = str(content)
        return hashlib.md5(content_str.encode()).hexdigest()
    
    def get_checkpoint(self) -> Any:
        """Get set of processed file paths"""
        return self.processed_files.copy()
    
    def close(self):
        """No cleanup needed for file system"""
        logger.info(f"Closed file connector for {self.root_path}")


class S3Connector(DataConnector):
    """Connector for AWS S3 buckets"""
    
    def __init__(self, source: DataSource):
        super().__init__(source)
        
        self.bucket_name = source.connection_params['bucket']
        self.prefix = source.connection_params.get('prefix', '')
        self.aws_access_key = source.connection_params.get('aws_access_key_id')
        self.aws_secret_key = source.connection_params.get('aws_secret_access_key')
        self.region = source.connection_params.get('region', 'us-east-1')
        
        self.s3_client = None
        self.processed_keys = set()
    
    def connect(self) -> bool:
        """Connect to S3"""
        try:
            import boto3
            
            self.s3_client = boto3.client(
                's3',
                aws_access_key_id=self.aws_access_key,
                aws_secret_access_key=self.aws_secret_key,
                region_name=self.region
            )
            
            # Test connection
            self.s3_client.head_bucket(Bucket=self.bucket_name)
            
            logger.info(f"Connected to S3 bucket: {self.bucket_name}")
            return True
        
        except Exception as e:
            logger.error(f"S3 connection failed: {e}")
            return False
    
    def extract(self, checkpoint: Optional[Any] = None) -> Iterator[IngestionRecord]:
        """Extract files from S3 bucket"""
        if not self.s3_client:
            raise RuntimeError("Not connected to S3")
        
        checkpoint = checkpoint or set()
        
        try:
            paginator = self.s3_client.get_paginator('list_objects_v2')
            pages = paginator.paginate(Bucket=self.bucket_name, Prefix=self.prefix)
            
            for page in pages:
                if 'Contents' not in page:
                    continue
                
                for obj in page['Contents']:
                    key = obj['Key']
                    
                    # Skip if already processed
                    if key in checkpoint:
                        continue
                    
                    try:
                        # Get object
                        response = self.s3_client.get_object(
                            Bucket=self.bucket_name,
                            Key=key
                        )
                        
                        # Read content
                        content_bytes = response['Body'].read()
                        
                        # Try to decode as text
                        try:
                            content_str = content_bytes.decode('utf-8')
                        except UnicodeDecodeError:
                            content_str = str(content_bytes)
                        
                        # Generate checksum
                        checksum = hashlib.md5(content_bytes).hexdigest()
                        
                        record = IngestionRecord(
                            record_id=checksum,
                            source_id=self.source.source_id,
                            content={
                                'type': 's3_object',
                                'content': content_str,
                                'key': key
                            },
                            metadata={
                                'bucket': self.bucket_name,
                                'key': key,
                                'size': obj['Size'],
                                'last_modified': obj['LastModified'].isoformat(),
                                'content_type': response.get('ContentType', 'unknown')
                            },
                            timestamp=obj['LastModified'],
                            checksum=checksum
                        )
                        
                        self.processed_keys.add(key)
                        yield record
                    
                    except Exception as e:
                        logger.error(f"Error processing S3 object {key}: {e}")
                        continue
        
        except Exception as e:
            logger.error(f"Error listing S3 objects: {e}")
            raise
    
    def get_checkpoint(self) -> Any:
        """Get set of processed S3 keys"""
        return self.processed_keys.copy()
    
    def close(self):
        """Close S3 connection"""
        self.s3_client = None
        logger.info(f"Closed S3 connector for bucket: {self.bucket_name}")