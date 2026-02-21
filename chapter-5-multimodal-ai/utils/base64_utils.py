"""
Base64 encoding/decoding utilities.
"""

import base64
from typing import Union
from PIL import Image
import io


class Base64Utils:
    """Utilities for base64 encoding/decoding"""
    
    @staticmethod
    def encode_file(file_path: str) -> str:
        """
        Encode file to base64 string
        
        Args:
            file_path: Path to file
            
        Returns:
            Base64 encoded string
        """
        with open(file_path, 'rb') as f:
            return base64.b64encode(f.read()).decode('utf-8')
    
    @staticmethod
    def decode_file(base64_string: str, output_path: str):
        """
        Decode base64 string to file
        
        Args:
            base64_string: Base64 encoded string
            output_path: Path to save decoded file
        """
        file_bytes = base64.b64decode(base64_string)
        
        with open(output_path, 'wb') as f:
            f.write(file_bytes)
    
    @staticmethod
    def encode_image(image: Union[str, Image.Image], format: str = 'PNG') -> str:
        """
        Encode image to base64
        
        Args:
            image: Image path or PIL Image
            format: Image format
            
        Returns:
            Base64 encoded string
        """
        if isinstance(image, str):
            image = Image.open(image)
        
        buffer = io.BytesIO()
        image.save(buffer, format=format)
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    @staticmethod
    def decode_image(base64_string: str) -> Image.Image:
        """
        Decode base64 string to image
        
        Args:
            base64_string: Base64 encoded string
            
        Returns:
            PIL Image
        """
        image_bytes = base64.b64decode(base64_string)
        return Image.open(io.BytesIO(image_bytes))