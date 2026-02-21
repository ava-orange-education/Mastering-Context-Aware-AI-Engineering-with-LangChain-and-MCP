"""
Base multimodal agent implementation with fusion strategies.
"""

from typing import Dict, Any, List, Optional, Union
from anthropic import Anthropic
import base64
from PIL import Image
import io
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MultiModalAgent:
    """Base class for multimodal AI agents"""
    
    def __init__(self, api_key: str, model: str = "claude-3-5-sonnet-20241022"):
        self.client = Anthropic(api_key=api_key)
        self.model = model
    
    def encode_image(self, image_path: str) -> tuple[str, str]:
        """
        Encode image to base64
        
        Args:
            image_path: Path to image file
            
        Returns:
            Tuple of (base64_string, media_type)
        """
        from PIL import Image
        
        # Determine media type from extension
        ext = image_path.lower().split('.')[-1]
        media_type_map = {
            'jpg': 'image/jpeg',
            'jpeg': 'image/jpeg',
            'png': 'image/png',
            'gif': 'image/gif',
            'webp': 'image/webp'
        }
        media_type = media_type_map.get(ext, 'image/jpeg')
        
        # Read and encode
        with open(image_path, 'rb') as f:
            image_data = base64.standard_b64encode(f.read()).decode('utf-8')
        
        return image_data, media_type
    
    def analyze_image(self, image_path: str, prompt: str) -> str:
        """
        Analyze an image with a text prompt
        
        Args:
            image_path: Path to image
            prompt: Analysis prompt
            
        Returns:
            Analysis result
        """
        image_data, media_type = self.encode_image(image_path)
        
        message = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": image_data
                            }
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }
            ]
        )
        
        return message.content[0].text
    
    def analyze_multiple_images(self, image_paths: List[str], prompt: str) -> str:
        """
        Analyze multiple images together
        
        Args:
            image_paths: List of image paths
            prompt: Analysis prompt
            
        Returns:
            Combined analysis
        """
        content = []
        
        # Add all images
        for image_path in image_paths:
            image_data, media_type = self.encode_image(image_path)
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": image_data
                }
            })
        
        # Add prompt
        content.append({
            "type": "text",
            "text": prompt
        })
        
        message = self.client.messages.create(
            model=self.model,
            max_tokens=2048,
            messages=[{"role": "user", "content": content}]
        )
        
        return message.content[0].text