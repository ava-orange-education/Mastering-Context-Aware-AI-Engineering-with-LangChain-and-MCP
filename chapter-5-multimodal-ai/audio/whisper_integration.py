"""
Whisper integration for audio transcription.
"""

from typing import Optional, Dict, Any, List
import logging
import torch

logger = logging.getLogger(__name__)


class WhisperIntegration:
    """Wrapper for Whisper model"""
    
    def __init__(self, model_size: str = "base"):
        """
        Initialize Whisper model
        
        Args:
            model_size: Model size (tiny, base, small, medium, large)
        """
        import whisper
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = whisper.load_model(model_size, device=self.device)
        
        logger.info(f"Loaded Whisper {model_size} model on {self.device}")
    
    def transcribe(self,
                   audio_path: str,
                   language: Optional[str] = None,
                   task: str = "transcribe") -> Dict[str, Any]:
        """
        Transcribe audio file
        
        Args:
            audio_path: Path to audio file
            language: Language code (e.g., 'en', 'es'), None for auto-detect
            task: 'transcribe' or 'translate' (to English)
            
        Returns:
            Transcription result with text and metadata
        """
        options = {
            "task": task,
            "temperature": 0.0,  # Deterministic output
        }
        
        if language:
            options["language"] = language
        
        result = self.model.transcribe(audio_path, **options)
        
        return {
            'text': result['text'],
            'language': result.get('language'),
            'segments': result.get('segments', [])
        }
    
    def transcribe_with_timestamps(self, audio_path: str) -> List[Dict[str, Any]]:
        """
        Transcribe with word-level timestamps
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            List of segments with timestamps
        """
        result = self.model.transcribe(
            audio_path,
            word_timestamps=True,
            temperature=0.0
        )
        
        segments = []
        for segment in result['segments']:
            segments.append({
                'start': segment['start'],
                'end': segment['end'],
                'text': segment['text']
            })
        
        return segments
    
    def detect_language(self, audio_path: str) -> Dict[str, float]:
        """
        Detect language in audio
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dictionary of language codes to probabilities
        """
        import whisper
        
        # Load audio
        audio = whisper.load_audio(audio_path)
        audio = whisper.pad_or_trim(audio)
        
        # Make log-Mel spectrogram
        mel = whisper.log_mel_spectrogram(audio).to(self.device)
        
        # Detect language
        _, probs = self.model.detect_language(mel)
        
        return {lang: float(prob) for lang, prob in probs.items()}