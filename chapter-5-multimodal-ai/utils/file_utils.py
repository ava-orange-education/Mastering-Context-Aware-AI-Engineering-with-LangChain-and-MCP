"""
File handling utilities.
"""

from pathlib import Path
from typing import List, Optional
import shutil
import logging

logger = logging.getLogger(__name__)


class FileUtils:
    """Utilities for file operations"""
    
    @staticmethod
    def ensure_directory(directory: str):
        """
        Ensure directory exists
        
        Args:
            directory: Directory path
        """
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    @staticmethod
    def get_file_extension(file_path: str) -> str:
        """
        Get file extension
        
        Args:
            file_path: Path to file
            
        Returns:
            File extension (lowercase, with dot)
        """
        return Path(file_path).suffix.lower()
    
    @staticmethod
    def list_files(directory: str, 
                  extensions: Optional[List[str]] = None,
                  recursive: bool = False) -> List[str]:
        """
        List files in directory
        
        Args:
            directory: Directory path
            extensions: Optional list of extensions to filter
            recursive: Whether to search recursively
            
        Returns:
            List of file paths
        """
        path = Path(directory)
        
        if not path.exists():
            return []
        
        if recursive:
            files = path.rglob('*')
        else:
            files = path.glob('*')
        
        files = [str(f) for f in files if f.is_file()]
        
        if extensions:
            extensions = [ext.lower() if ext.startswith('.') else f'.{ext.lower()}' 
                         for ext in extensions]
            files = [f for f in files if Path(f).suffix.lower() in extensions]
        
        return files
    
    @staticmethod
    def copy_file(source: str, destination: str):
        """
        Copy file
        
        Args:
            source: Source file path
            destination: Destination file path
        """
        shutil.copy2(source, destination)
    
    @staticmethod
    def move_file(source: str, destination: str):
        """
        Move file
        
        Args:
            source: Source file path
            destination: Destination file path
        """
        shutil.move(source, destination)
    
    @staticmethod
    def delete_file(file_path: str):
        """
        Delete file
        
        Args:
            file_path: Path to file
        """
        try:
            Path(file_path).unlink()
        except Exception as e:
            logger.error(f"Error deleting file {file_path}: {e}")
    
    @staticmethod
    def get_file_size(file_path: str) -> int:
        """
        Get file size in bytes
        
        Args:
            file_path: Path to file
            
        Returns:
            File size in bytes
        """
        return Path(file_path).stat().st_size