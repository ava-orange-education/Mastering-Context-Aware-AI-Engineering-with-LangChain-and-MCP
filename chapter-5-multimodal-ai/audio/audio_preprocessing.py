"""
Audio preprocessing utilities.
"""

from typing import Tuple, Optional
import numpy as np


class AudioPreprocessor:
    """Utilities for audio preprocessing"""
    
    @staticmethod
    def load_audio(audio_path: str, sample_rate: int = 16000) -> np.ndarray:
        """
        Load audio file
        
        Args:
            audio_path: Path to audio file
            sample_rate: Target sample rate
            
        Returns:
            Audio data as numpy array
        """
        import librosa
        
        audio, sr = librosa.load(audio_path, sr=sample_rate)
        return audio
    
    @staticmethod
    def resample_audio(audio: np.ndarray, 
                      orig_sr: int, 
                      target_sr: int) -> np.ndarray:
        """
        Resample audio to different sample rate
        
        Args:
            audio: Audio data
            orig_sr: Original sample rate
            target_sr: Target sample rate
            
        Returns:
            Resampled audio
        """
        import librosa
        
        return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)
    
    @staticmethod
    def trim_silence(audio: np.ndarray, 
                    top_db: int = 20) -> Tuple[np.ndarray, Tuple[int, int]]:
        """
        Trim silence from audio
        
        Args:
            audio: Audio data
            top_db: Threshold in decibels below reference
            
        Returns:
            Tuple of (trimmed_audio, (start_index, end_index))
        """
        import librosa
        
        trimmed, index = librosa.effects.trim(audio, top_db=top_db)
        return trimmed, index
    
    @staticmethod
    def normalize_audio(audio: np.ndarray) -> np.ndarray:
        """
        Normalize audio to [-1, 1] range
        
        Args:
            audio: Audio data
            
        Returns:
            Normalized audio
        """
        max_val = np.abs(audio).max()
        if max_val > 0:
            return audio / max_val
        return audio
    
    @staticmethod
    def compute_mel_spectrogram(audio: np.ndarray,
                               sample_rate: int = 16000,
                               n_mels: int = 80) -> np.ndarray:
        """
        Compute mel spectrogram
        
        Args:
            audio: Audio data
            sample_rate: Sample rate
            n_mels: Number of mel bands
            
        Returns:
            Mel spectrogram
        """
        import librosa
        
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=sample_rate,
            n_mels=n_mels
        )
        
        # Convert to log scale
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        
        return log_mel_spec