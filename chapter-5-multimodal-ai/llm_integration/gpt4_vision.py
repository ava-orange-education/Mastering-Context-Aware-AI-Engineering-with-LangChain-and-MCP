"""
GPT-4 Vision integration.
"""

from typing import List, Dict, Any
import base64
import requests


class GPT4Vision:
    """GPT-4 Vision capabilities"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.model = "gpt-4-vision-preview"
    
    def encode_image(self, image_path: str) -> str:
        """Encode image to base64"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    def analyze_image(self, 
                     image_path: str,
                     prompt: str,
                     max_tokens: int = 1024) -> str:
        """
        Analyze image with GPT-4 Vision
        
        Args:
            image_path: Path to image
            prompt: Analysis prompt
            max_tokens: Maximum response tokens
            
        Returns:
            Analysis result
        """
        base64_image = self.encode_image(image_path)
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": max_tokens
        }
        
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=payload
        )
        
        return response.json()["choices"][0]["message"]["content"]
    
    def analyze_multiple_images(self,
                                image_paths: List[str],
                                prompt: str) -> str:
        """
        Analyze multiple images together
        
        Args:
            image_paths: List of image paths
            prompt: Analysis prompt
            
        Returns:
            Combined analysis
        """
        content = [{"type": "text", "text": prompt}]
        
        for image_path in image_paths:
            base64_image = self.encode_image(image_path)
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                }
            })
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": 2048
        }
        
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=payload
        )
        
        return response.json()["choices"][0]["message"]["content"]