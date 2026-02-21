"""
Fusion strategies for combining multimodal information.
"""

from typing import Dict, Any, List
import numpy as np


class FusionStrategy:
    """Base class for fusion strategies"""
    
    def fuse(self, modalities: Dict[str, Any]) -> Any:
        """Fuse information from multiple modalities"""
        raise NotImplementedError


class EarlyFusion(FusionStrategy):
    """Concatenate features from different modalities early"""
    
    def fuse(self, modalities: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Concatenate features from all modalities
        
        Args:
            modalities: Dict mapping modality name to feature vector
            
        Returns:
            Concatenated feature vector
        """
        features = []
        for modality_name in sorted(modalities.keys()):
            features.append(modalities[modality_name])
        
        return np.concatenate(features)


class LateFusion(FusionStrategy):
    """Process modalities independently and combine predictions"""
    
    def __init__(self, weights: Optional[Dict[str, float]] = None):
        self.weights = weights or {}
    
    def fuse(self, modality_predictions: Dict[str, float]) -> float:
        """
        Weighted average of predictions from each modality
        
        Args:
            modality_predictions: Dict mapping modality to prediction score
            
        Returns:
            Fused prediction score
        """
        if not self.weights:
            # Equal weights
            return sum(modality_predictions.values()) / len(modality_predictions)
        
        total_weight = sum(self.weights.values())
        weighted_sum = sum(
            modality_predictions[mod] * self.weights.get(mod, 1.0)
            for mod in modality_predictions
        )
        
        return weighted_sum / total_weight


class AttentionFusion(FusionStrategy):
    """Use attention mechanism to weight modalities dynamically"""
    
    def __init__(self, embedding_dim: int = 512):
        self.embedding_dim = embedding_dim
    
    def compute_attention_weights(self, 
                                  features: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Compute attention weights for each modality
        
        Args:
            features: Feature vectors for each modality
            
        Returns:
            Attention weights (sum to 1.0)
        """
        # Simplified attention: use feature magnitude
        magnitudes = {
            name: np.linalg.norm(feat)
            for name, feat in features.items()
        }
        
        total = sum(magnitudes.values())
        weights = {
            name: mag / total
            for name, mag in magnitudes.items()
        }
        
        return weights
    
    def fuse(self, features: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Fuse features using attention weights
        
        Args:
            features: Feature vectors for each modality
            
        Returns:
            Attention-weighted fused features
        """
        weights = self.compute_attention_weights(features)
        
        fused = np.zeros_like(list(features.values())[0])
        for modality, feat in features.items():
            fused += weights[modality] * feat
        
        return fused