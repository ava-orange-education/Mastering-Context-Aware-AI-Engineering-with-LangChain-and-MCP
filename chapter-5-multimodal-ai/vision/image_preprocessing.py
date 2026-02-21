"""
Image preprocessing utilities.
"""

from typing import Tuple, Optional, Union
from PIL import Image
import numpy as np
import base64
import io


class ImagePreprocessor:
    """Utilities for image preprocessing"""
    
    @staticmethod
    def load_image(image_path: str) -> Image.Image:
        """
        Load image from path
        
        Args:
            image_path: Path to image file
            
        Returns:
            PIL Image object
        """
        return Image.open(image_path).convert('RGB')
    
    @staticmethod
    def resize_image(image: Image.Image, 
                    max_size: Tuple[int, int] = (1024, 1024),
                    maintain_aspect_ratio: bool = True) -> Image.Image:
        """
        Resize image
        
        Args:
            image: PIL Image
            max_size: Maximum (width, height)
            maintain_aspect_ratio: Whether to maintain aspect ratio
            
        Returns:
            Resized image
        """
        if maintain_aspect_ratio:
            image.thumbnail(max_size, Image.Resampling.LANCZOS)
            return image
        else:
            return image.resize(max_size, Image.Resampling.LANCZOS)
    
    @staticmethod
    def encode_image_to_base64(image: Union[str, Image.Image]) -> Tuple[str, str]:
        """
        Encode image to base64 string
        
        Args:
            image: Image path or PIL Image
            
        Returns:
            Tuple of (base64_string, media_type)
        """
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        
        # Convert to bytes
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        image_bytes = buffer.getvalue()
        
        # Encode to base64
        base64_string = base64.b64encode(image_bytes).decode('utf-8')
        
        return base64_string, 'image/png'
    
    @staticmethod
    def decode_base64_to_image(base64_string: str) -> Image.Image:
        """
        Decode base64 string to image
        
        Args:
            base64_string: Base64 encoded image
            
        Returns:
            PIL Image
        """
        image_bytes = base64.b64decode(base64_string)
        return Image.open(io.BytesIO(image_bytes))
    
    @staticmethod
    def crop_image(image: Image.Image, bbox: Tuple[int, int, int, int]) -> Image.Image:
        """
        Crop image to bounding box
        
        Args:
            image: PIL Image
            bbox: Bounding box (x1, y1, x2, y2)
            
        Returns:
            Cropped image
        """
        return image.crop(bbox)
    
    @staticmethod
    def normalize_image(image: np.ndarray,
                       mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
                       std: Tuple[float, float, float] = (0.229, 0.224, 0.225)) -> np.ndarray:
        """
        Normalize image for neural network input
        
        Args:
            image: Image as numpy array
            mean: Mean values for normalization
            std: Standard deviation values
            
        Returns:
            Normalized image
        """
        image = image.astype(np.float32) / 255.0
        image = (image - np.array(mean)) / np.array(std)
        return image