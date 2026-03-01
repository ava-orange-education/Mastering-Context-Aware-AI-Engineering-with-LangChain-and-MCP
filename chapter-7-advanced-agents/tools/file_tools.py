"""
File manipulation tools for reading, writing, and managing files.
"""

from typing import Dict, Any, List
from .tool_base import Tool, ToolParameter
import os
import logging

logger = logging.getLogger(__name__)


class ReadFileTool(Tool):
    """Tool for reading file contents"""
    
    def __init__(self, allowed_paths: Optional[List[str]] = None):
        super().__init__(
            name="read_file",
            description="Read contents of a file"
        )
        
        self.parameters = [
            ToolParameter(
                name="filepath",
                type="string",
                description="Path to file to read"
            )
        ]
        
        self.allowed_paths = allowed_paths or []
    
    def execute(self, input_data: Dict[str, Any]) -> Any:
        """Read file"""
        self.validate_input(input_data)
        
        filepath = input_data['filepath']
        
        # Security check
        if self.allowed_paths and not any(filepath.startswith(p) for p in self.allowed_paths):
            return {"error": f"Access denied: {filepath} not in allowed paths"}
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            logger.info(f"Read file: {filepath} ({len(content)} chars)")
            
            return {
                "filepath": filepath,
                "content": content,
                "size": len(content)
            }
        
        except Exception as e:
            logger.error(f"Failed to read {filepath}: {e}")
            return {"error": str(e)}


class WriteFileTool(Tool):
    """Tool for writing content to files"""
    
    def __init__(self, allowed_paths: Optional[List[str]] = None):
        super().__init__(
            name="write_file",
            description="Write content to a file"
        )
        
        self.parameters = [
            ToolParameter(
                name="filepath",
                type="string",
                description="Path where to write file"
            ),
            ToolParameter(
                name="content",
                type="string",
                description="Content to write"
            ),
            ToolParameter(
                name="mode",
                type="string",
                description="Write mode: 'w' (overwrite) or 'a' (append)",
                required=False,
                default="w"
            )
        ]
        
        self.allowed_paths = allowed_paths or []
    
    def execute(self, input_data: Dict[str, Any]) -> Any:
        """Write to file"""
        self.validate_input(input_data)
        
        filepath = input_data['filepath']
        content = input_data['content']
        mode = input_data.get('mode', 'w')
        
        # Security check
        if self.allowed_paths and not any(filepath.startswith(p) for p in self.allowed_paths):
            return {"error": f"Access denied: {filepath} not in allowed paths"}
        
        try:
            # Create directory if needed
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            with open(filepath, mode, encoding='utf-8') as f:
                f.write(content)
            
            logger.info(f"Wrote to file: {filepath} ({len(content)} chars)")
            
            return {
                "filepath": filepath,
                "bytes_written": len(content.encode('utf-8')),
                "mode": mode
            }
        
        except Exception as e:
            logger.error(f"Failed to write {filepath}: {e}")
            return {"error": str(e)}


class ListDirectoryTool(Tool):
    """Tool for listing directory contents"""
    
    def __init__(self, allowed_paths: Optional[List[str]] = None):
        super().__init__(
            name="list_directory",
            description="List files and directories in a path"
        )
        
        self.parameters = [
            ToolParameter(
                name="path",
                type="string",
                description="Directory path to list"
            ),
            ToolParameter(
                name="recursive",
                type="boolean",
                description="Whether to list recursively",
                required=False,
                default=False
            )
        ]
        
        self.allowed_paths = allowed_paths or []
    
    def execute(self, input_data: Dict[str, Any]) -> Any:
        """List directory"""
        self.validate_input(input_data)
        
        path = input_data['path']
        recursive = input_data.get('recursive', False)
        
        # Security check
        if self.allowed_paths and not any(path.startswith(p) for p in self.allowed_paths):
            return {"error": f"Access denied: {path} not in allowed paths"}
        
        try:
            if recursive:
                files = []
                for root, dirs, filenames in os.walk(path):
                    for filename in filenames:
                        files.append(os.path.join(root, filename))
            else:
                files = [os.path.join(path, f) for f in os.listdir(path)]
            
            logger.info(f"Listed directory: {path} ({len(files)} items)")
            
            return {
                "path": path,
                "files": files,
                "count": len(files)
            }
        
        except Exception as e:
            logger.error(f"Failed to list {path}: {e}")
            return {"error": str(e)}


class FileOperationsTool(Tool):
    """Tool for various file operations"""
    
    def __init__(self, allowed_paths: Optional[List[str]] = None):
        super().__init__(
            name="file_operations",
            description="Perform file operations: copy, move, delete"
        )
        
        self.parameters = [
            ToolParameter(
                name="operation",
                type="string",
                description="Operation: 'copy', 'move', 'delete'"
            ),
            ToolParameter(
                name="source",
                type="string",
                description="Source file path"
            ),
            ToolParameter(
                name="destination",
                type="string",
                description="Destination path (not needed for delete)",
                required=False
            )
        ]
        
        self.allowed_paths = allowed_paths or []
    
    def execute(self, input_data: Dict[str, Any]) -> Any:
        """Execute file operation"""
        self.validate_input(input_data)
        
        operation = input_data['operation']
        source = input_data['source']
        destination = input_data.get('destination')
        
        # Security checks
        if self.allowed_paths:
            if not any(source.startswith(p) for p in self.allowed_paths):
                return {"error": f"Access denied: {source}"}
            
            if destination and not any(destination.startswith(p) for p in self.allowed_paths):
                return {"error": f"Access denied: {destination}"}
        
        try:
            import shutil
            
            if operation == "copy":
                shutil.copy2(source, destination)
                logger.info(f"Copied {source} to {destination}")
                return {"operation": "copy", "source": source, "destination": destination}
            
            elif operation == "move":
                shutil.move(source, destination)
                logger.info(f"Moved {source} to {destination}")
                return {"operation": "move", "source": source, "destination": destination}
            
            elif operation == "delete":
                os.remove(source)
                logger.info(f"Deleted {source}")
                return {"operation": "delete", "source": source}
            
            else:
                return {"error": f"Unknown operation: {operation}"}
        
        except Exception as e:
            logger.error(f"File operation failed: {e}")
            return {"error": str(e)}