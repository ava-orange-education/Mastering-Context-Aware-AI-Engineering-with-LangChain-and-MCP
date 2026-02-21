"""
Specialized vision agent for MCP system.
"""

from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)


class VisionAgent:
    """Agent specialized in vision tasks"""
    
    def __init__(self, api_key: str):
        from vision import CLIPIntegration, BLIPIntegration
        from llm_integration import ClaudeMultimodal
        
        self.clip = CLIPIntegration()
        self.blip = BLIPIntegration()
        self.claude = ClaudeMultimodal(api_key)
        
        self.capabilities = [
            'image_classification',
            'image_captioning',
            'visual_qa',
            'object_detection',
            'image_search'
        ]
    
    def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle vision-related request
        
        Args:
            request: Request dictionary with 'task' and parameters
            
        Returns:
            Result dictionary
        """
        task = request.get('task')
        
        if task == 'image_classification':
            return self._classify_image(request)
        elif task == 'image_captioning':
            return self._caption_image(request)
        elif task == 'visual_qa':
            return self._answer_visual_question(request)
        elif task == 'image_search':
            return self._search_images(request)
        else:
            return {'error': f'Unknown task: {task}'}
    
    def _classify_image(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Classify image"""
        image_path = request['image_path']
        labels = request.get('labels', ['dog', 'cat', 'bird', 'car'])
        
        results = self.clip.zero_shot_classification(image_path, labels)
        
        return {
            'task': 'image_classification',
            'image_path': image_path,
            'results': results
        }
    
    def _caption_image(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Generate image caption"""
        image_path = request['image_path']
        
        caption = self.blip.generate_caption(image_path)
        
        return {
            'task': 'image_captioning',
            'image_path': image_path,
            'caption': caption
        }
    
    def _answer_visual_question(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Answer question about image"""
        image_path = request['image_path']
        question = request['question']
        
        answer = self.claude.analyze_image(image_path, question)
        
        return {
            'task': 'visual_qa',
            'image_path': image_path,
            'question': question,
            'answer': answer
        }
    
    def _search_images(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Search images by text query"""
        image_paths = request['image_paths']
        query = request['query']
        top_k = request.get('top_k', 5)
        
        results = self.clip.semantic_image_search(image_paths, query, top_k)
        
        return {
            'task': 'image_search',
            'query': query,
            'results': results
        }