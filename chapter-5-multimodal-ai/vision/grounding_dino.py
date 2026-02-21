"""
Grounding DINO for open-vocabulary object detection.
"""

from typing import List, Tuple, Dict, Any
import logging
from PIL import Image
import torch
import numpy as np

logger = logging.getLogger(__name__)


class GroundingDINOIntegration:
    """Wrapper for Grounding DINO model"""
    
    def __init__(self):
        """Initialize Grounding DINO model"""
        try:
            from groundingdino.util.inference import load_model, predict
            
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            
            # Load model
            config_path = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
            checkpoint_path = "weights/groundingdino_swint_ogc.pth"
            
            self.model = load_model(config_path, checkpoint_path)
            self.predict_fn = predict
            
            logger.info(f"Loaded Grounding DINO on {self.device}")
            
        except ImportError:
            logger.error("Grounding DINO not installed. Install from: https://github.com/IDEA-Research/GroundingDINO")
            raise
    
    def detect_objects(self,
                      image_path: str,
                      text_prompt: str,
                      box_threshold: float = 0.35,
                      text_threshold: float = 0.25) -> List[Dict[str, Any]]:
        """
        Detect objects based on text prompt
        
        Args:
            image_path: Path to image
            text_prompt: Text description of objects to detect
            box_threshold: Confidence threshold for bounding boxes
            text_threshold: Confidence threshold for text matching
            
        Returns:
            List of detections with boxes, labels, and scores
        """
        image = Image.open(image_path).convert('RGB')
        
        # Predict
        boxes, logits, phrases = self.predict_fn(
            model=self.model,
            image=image,
            caption=text_prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold
        )
        
        # Convert to list of detections
        detections = []
        h, w = image.size[1], image.size[0]
        
        for box, logit, phrase in zip(boxes, logits, phrases):
            # Convert normalized coordinates to pixel coordinates
            x1, y1, x2, y2 = box.tolist()
            x1, x2 = int(x1 * w), int(x2 * w)
            y1, y2 = int(y1 * h), int(y2 * h)
            
            detections.append({
                'bbox': [x1, y1, x2, y2],
                'label': phrase,
                'score': float(logit)
            })
        
        return detections
    
    def detect_and_annotate(self,
                           image_path: str,
                           text_prompt: str,
                           output_path: str,
                           box_threshold: float = 0.35) -> List[Dict[str, Any]]:
        """
        Detect objects and save annotated image
        
        Args:
            image_path: Path to input image
            text_prompt: Text description of objects
            output_path: Path to save annotated image
            box_threshold: Confidence threshold
            
        Returns:
            List of detections
        """
        from PIL import ImageDraw, ImageFont
        
        detections = self.detect_objects(image_path, text_prompt, box_threshold)
        
        # Load image
        image = Image.open(image_path).convert('RGB')
        draw = ImageDraw.Draw(image)
        
        # Try to load a font, fallback to default
        try:
            font = ImageFont.truetype("arial.ttf", 15)
        except:
            font = ImageFont.load_default()
        
        # Draw boxes and labels
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            
            # Draw rectangle
            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
            
            # Draw label
            label_text = f"{det['label']}: {det['score']:.2f}"
            draw.text((x1, y1 - 20), label_text, fill="red", font=font)
        
        # Save
        image.save(output_path)
        logger.info(f"Annotated image saved to {output_path}")
        
        return detections