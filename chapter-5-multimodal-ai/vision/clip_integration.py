"""
CLIP (Contrastive Language-Image Pre-training) integration.
"""

from typing import List, Tuple, Dict, Any
import logging
import torch
from PIL import Image
import numpy as np

logger = logging.getLogger(__name__)


class CLIPIntegration:
    """Wrapper for CLIP model"""
    
    def __init__(self, model_name: str = "ViT-B/32"):
        """
        Initialize CLIP model
        
        Args:
            model_name: CLIP model variant
        """
        import clip
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        
        logger.info(f"Loaded CLIP model {model_name} on {self.device}")
    
    def encode_image(self, image_path: str) -> np.ndarray:
        """
        Encode image to feature vector
        
        Args:
            image_path: Path to image
            
        Returns:
            Image embedding
        """
        image = Image.open(image_path).convert('RGB')
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            image_features = self.model.encode_image(image_input)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        return image_features.cpu().numpy()[0]
    
    def encode_text(self, text: str) -> np.ndarray:
        """
        Encode text to feature vector
        
        Args:
            text: Text to encode
            
        Returns:
            Text embedding
        """
        import clip
        
        text_input = clip.tokenize([text]).to(self.device)
        
        with torch.no_grad():
            text_features = self.model.encode_text(text_input)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        return text_features.cpu().numpy()[0]
    
    def zero_shot_classification(self, 
                                 image_path: str,
                                 candidate_labels: List[str]) -> List[Tuple[str, float]]:
        """
        Zero-shot image classification
        
        Args:
            image_path: Path to image
            candidate_labels: List of possible labels
            
        Returns:
            List of (label, probability) tuples sorted by probability
        """
        import clip
        
        # Encode image
        image = Image.open(image_path).convert('RGB')
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)
        
        # Encode text labels
        text_inputs = clip.tokenize([f"a photo of a {label}" for label in candidate_labels]).to(self.device)
        
        with torch.no_grad():
            image_features = self.model.encode_image(image_input)
            text_features = self.model.encode_text(text_inputs)
            
            # Normalize features
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            # Calculate similarity
            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        
        # Create results
        results = [
            (label, float(prob))
            for label, prob in zip(candidate_labels, similarity[0])
        ]
        
        return sorted(results, key=lambda x: x[1], reverse=True)
    
    def semantic_image_search(self,
                             image_paths: List[str],
                             query: str,
                             top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Search images using text query
        
        Args:
            image_paths: List of image paths to search
            query: Text search query
            top_k: Number of results to return
            
        Returns:
            List of (image_path, similarity_score) tuples
        """
        # Encode query
        query_features = self.encode_text(query)
        
        # Encode all images
        similarities = []
        for image_path in image_paths:
            try:
                image_features = self.encode_image(image_path)
                similarity = float(np.dot(query_features, image_features))
                similarities.append((image_path, similarity))
            except Exception as e:
                logger.error(f"Error processing {image_path}: {e}")
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]