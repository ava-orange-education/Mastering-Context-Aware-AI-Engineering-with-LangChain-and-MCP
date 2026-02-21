"""
Input preprocessing for personal assistant.
"""

from typing import Dict, Any, Optional
from PIL import Image
import logging

logger = logging.getLogger(__name__)


class InputPreprocessor:
    """Preprocess inputs before sending to models"""
    
    @staticmethod
    def preprocess_image(image_path: str, 
                        max_size: tuple = (1024, 1024),
                        quality: int = 85) -> str:
        """
        Preprocess image (resize, compress)
        
        Args:
            image_path: Path to original image
            max_size: Maximum dimensions
            quality: JPEG quality (1-100)
            
        Returns:
            Path to preprocessed image
        """
        img = Image.open(image_path)
        
        # Resize if needed
        if img.size[0] > max_size[0] or img.size[1] > max_size[1]:
            img.thumbnail(max_size, Image.Resampling.LANCZOS)
        
        # Save preprocessed image
        output_path = image_path.replace('.', '_preprocessed.')
        img.save(output_path, quality=quality, optimize=True)
        
        return output_path
    
    @staticmethod
    def preprocess_audio(audio_path: str,
                        target_sample_rate: int = 16000) -> str:
        """
        Preprocess audio (resample, normalize)
        
        Args:
            audio_path: Path to original audio
            target_sample_rate: Target sample rate
            
        Returns:
            Path to preprocessed audio
        """
        try:
            import librosa
            import soundfile as sf
            
            # Load audio
            audio, sr = librosa.load(audio_path, sr=target_sample_rate)
            
            # Normalize
            audio = audio / (max(abs(audio)) + 1e-8)
            
            # Save preprocessed audio
            output_path = audio_path.replace('.', '_preprocessed.')
            sf.write(output_path, audio, target_sample_rate)
            
            return output_path
            
        except ImportError:
            logger.warning("librosa/soundfile not available, skipping audio preprocessing")
            return audio_path
    
    @staticmethod
    def preprocess_text(text: str) -> str:
        """
        Preprocess text (clean, normalize)
        
        Args:
            text: Input text
            
        Returns:
            Preprocessed text
        """
        # Remove excessive whitespace
        text = ' '.join(text.split())
        
        # Remove null characters
        text = text.replace('\x00', '')
        
        return text.strip()
    
    @staticmethod
    def preprocess_request(request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Preprocess entire request
        
        Args:
            request: Original request
            
        Returns:
            Preprocessed request
        """
        preprocessed = request.copy()
        
        # Preprocess image if present
        if 'image_path' in request:
            try:
                preprocessed['image_path'] = InputPreprocessor.preprocess_image(
                    request['image_path']
                )
            except Exception as e:
                logger.error(f"Image preprocessing failed: {e}")
        
        # Preprocess audio if present
        if 'audio_path' in request:
            try:
                preprocessed['audio_path'] = InputPreprocessor.preprocess_audio(
                    request['audio_path']
                )
            except Exception as e:
                logger.error(f"Audio preprocessing failed: {e}")
        
        # Preprocess text fields
        for key in ['query', 'prompt', 'question']:
            if key in request and isinstance(request[key], str):
                preprocessed[key] = InputPreprocessor.preprocess_text(request[key])
        
        return preprocessed