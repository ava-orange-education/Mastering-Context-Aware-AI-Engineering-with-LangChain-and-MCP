"""
Tests for audio components.
"""

import unittest
import sys
sys.path.append('..')

from audio.whisper_integration import WhisperIntegration
from audio.audio_preprocessing import AudioPreprocessor


class TestAudio(unittest.TestCase):
    """Test audio processing components"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures"""
        cls.whisper = WhisperIntegration(model_size="tiny")  # Use tiny for faster tests
        cls.preprocessor = AudioPreprocessor()
        cls.test_audio = '../data/test_data/test_audio.mp3'
    
    def test_transcription(self):
        """Test audio transcription"""
        result = self.whisper.transcribe(self.test_audio)
        
        self.assertIn('text', result)
        self.assertIsInstance(result['text'], str)
        self.assertGreater(len(result['text']), 0)
    
    def test_language_detection(self):
        """Test language detection"""
        languages = self.whisper.detect_language(self.test_audio)
        
        self.assertIsInstance(languages, dict)
        self.assertGreater(len(languages), 0)
        
        # Check probabilities sum to 1
        total_prob = sum(languages.values())
        self.assertAlmostEqual(total_prob, 1.0, places=2)
    
    def test_audio_normalization(self):
        """Test audio normalization"""
        import librosa
        
        audio = librosa.load(self.test_audio, sr=16000)[0]
        normalized = self.preprocessor.normalize_audio(audio)
        
        # Normalized audio should be in [-1, 1] range
        self.assertLessEqual(normalized.max(), 1.0)
        self.assertGreaterEqual(normalized.min(), -1.0)


if __name__ == '__main__':
    unittest.main()