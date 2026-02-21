"""
Claude API integration with vision capabilities.
"""

from typing import List, Dict, Any, Optional
from anthropic import Anthropic
import base64


class ClaudeMultimodal:
    """Claude multimodal capabilities"""
    
    def __init__(self, api_key: str):
        self.client = Anthropic(api_key=api_key)
        self.model = "claude-3-5-sonnet-20241022"
    
    def analyze_image(self, 
                     image_path: str, 
                     prompt: str,
                     max_tokens: int = 1024) -> str:
        """
        Analyze image with text prompt
        
        Args:
            image_path: Path to image
            prompt: Analysis prompt
            max_tokens: Maximum response tokens
            
        Returns:
            Analysis result
        """
        # Read and encode image
        with open(image_path, 'rb') as f:
            image_data = base64.standard_b64encode(f.read()).decode('utf-8')
        
        # Determine media type
        ext = image_path.lower().split('.')[-1]
        media_type = f"image/{ext}" if ext != 'jpg' else "image/jpeg"
        
        message = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=[{
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
            }]
        )
        
        return message.content[0].text
    
    def compare_images(self, 
                      image_paths: List[str],
                      prompt: str) -> str:
        """
        Compare multiple images
        
        Args:
            image_paths: List of image paths
            prompt: Comparison prompt
            
        Returns:
            Comparison result
        """
        content = []
        
        # Add all images
        for image_path in image_paths:
            with open(image_path, 'rb') as f:
                image_data = base64.standard_b64encode(f.read()).decode('utf-8')
            
            ext = image_path.lower().split('.')[-1]
            media_type = f"image/{ext}" if ext != 'jpg' else "image/jpeg"
            
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
    
    def document_analysis(self, 
                         document_path: str,
                         questions: List[str]) -> List[str]:
        """
        Analyze document and answer questions
        
        Args:
            document_path: Path to document (PDF/image)
            questions: List of questions
            
        Returns:
            List of answers
        """
        answers = []
        
        for question in questions:
            answer = self.analyze_image(document_path, question)
            answers.append(answer)
        
        return answers