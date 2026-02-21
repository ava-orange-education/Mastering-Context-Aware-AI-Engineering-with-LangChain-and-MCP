"""
BLIP/BLIP-2 integration for image captioning and VQA.
"""

from typing import Optional, List
import logging
from PIL import Image
import torch

logger = logging.getLogger(__name__)


class BLIPIntegration:
    """Wrapper for BLIP model"""
    
    def __init__(self, model_type: str = "blip-base"):
        """
        Initialize BLIP model

        Args:
            model_type: 'blip-base' or 'blip-large'
        """
        import warnings
        from transformers import BlipProcessor, BlipForConditionalGeneration

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if model_type == "blip-base":
            model_name = "Salesforce/blip-image-captioning-base"
        else:
            model_name = "Salesforce/blip-image-captioning-large"

        # Silence transformers warnings (tied weights, load report)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            try:
                import transformers
                transformers.logging.set_verbosity_error()
            except AttributeError:
                pass
            self.processor = BlipProcessor.from_pretrained(model_name)
            self.model = BlipForConditionalGeneration.from_pretrained(model_name).to(self.device)
        
        logger.info(f"Loaded BLIP model {model_type} on {self.device}")
    
    def generate_caption(self, 
                        image_path: str,
                        max_length: int = 50,
                        num_beams: int = 5) -> str:
        """
        Generate caption for image
        
        Args:
            image_path: Path to image
            max_length: Maximum caption length
            num_beams: Number of beams for beam search
            
        Returns:
            Generated caption
        """
        image = Image.open(image_path).convert('RGB')
        
        # Unconditional image captioning
        inputs = self.processor(image, return_tensors="pt").to(self.device)
        
        outputs = self.model.generate(
            **inputs,
            max_length=max_length,
            num_beams=num_beams
        )
        
        caption = self.processor.decode(outputs[0], skip_special_tokens=True)
        
        return caption
    
    def conditional_caption(self,
                           image_path: str,
                           prompt: str,
                           max_length: int = 50) -> str:
        """
        Generate caption conditioned on a text prompt
        
        Args:
            image_path: Path to image
            prompt: Conditioning text
            max_length: Maximum caption length
            
        Returns:
            Generated caption
        """
        image = Image.open(image_path).convert('RGB')
        
        inputs = self.processor(image, prompt, return_tensors="pt").to(self.device)
        
        outputs = self.model.generate(**inputs, max_length=max_length)
        
        caption = self.processor.decode(outputs[0], skip_special_tokens=True)
        
        return caption


class BLIP2Integration:
    """Wrapper for BLIP-2 model with Q-Former"""
    
    def __init__(self, model_type: str = "blip2-opt-2.7b"):
        """
        Initialize BLIP-2 model
        
        Args:
            model_type: Model variant
        """
        from transformers import Blip2Processor, Blip2ForConditionalGeneration
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        model_name = f"Salesforce/{model_type}"
        
        self.processor = Blip2Processor.from_pretrained(model_name)
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        ).to(self.device)
        
        logger.info(f"Loaded BLIP-2 model {model_type} on {self.device}")
    
    def answer_question(self,
                       image_path: str,
                       question: str,
                       max_length: int = 50) -> str:
        """
        Visual Question Answering
        
        Args:
            image_path: Path to image
            question: Question about the image
            max_length: Maximum answer length
            
        Returns:
            Answer to the question
        """
        image = Image.open(image_path).convert('RGB')
        
        prompt = f"Question: {question} Answer:"
        
        inputs = self.processor(image, prompt, return_tensors="pt").to(self.device)
        
        outputs = self.model.generate(**inputs, max_length=max_length)
        
        answer = self.processor.decode(outputs[0], skip_special_tokens=True)
        
        return answer
    
    def batch_vqa(self,
                  image_path: str,
                  questions: List[str]) -> List[str]:
        """
        Answer multiple questions about an image
        
        Args:
            image_path: Path to image
            questions: List of questions
            
        Returns:
            List of answers
        """
        answers = []
        
        for question in questions:
            answer = self.answer_question(image_path, question)
            answers.append(answer)
        
        return answers