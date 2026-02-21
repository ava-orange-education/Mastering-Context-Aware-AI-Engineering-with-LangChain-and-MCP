"""
Tests for vision components.
"""

import unittest
import sys
sys.path.append('..')

from vision.clip_integration import CLIPIntegration
from vision.image_preprocessing import ImagePreprocessor


class TestVision(unittest.TestCase):
    """Test vision processing components"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures"""
        cls.clip = CLIPIntegration()
        cls.preprocessor = ImagePreprocessor()
        cls.test_image = '../data/test_data/test_image.jpg'
    
    def test_clip_encoding(self):
        """Test CLIP image encoding"""
        embedding = self.clip.encode_image(self.test_image)
        
        self.assertIsNotNone(embedding)
        self.assertEqual(len(embedding.shape), 1)  # Should be 1D vector
        self.assertGreater(len(embedding), 0)
    
    def test_zero_shot_classification(self):
        """Test zero-shot classification"""
        labels = ['dog', 'cat', 'bird']
        results = self.clip.zero_shot_classification(self.test_image, labels)
        
        self.assertEqual(len(results), len(labels))
        self.assertIsInstance(results[0], tuple)
        self.assertIsInstance(results[0][0], str)  # Label
        self.assertIsInstance(results[0][1], float)  # Probability
        
        # Probabilities should sum to ~1
        total_prob = sum(prob for _, prob in results)
        self.assertAlmostEqual(total_prob, 1.0, places=2)
    
    def test_image_preprocessing(self):
        """Test image preprocessing"""
        preprocessed = self.preprocessor.preprocess_image(self.test_image)
        
        self.assertIsNotNone(preprocessed)
        self.assertTrue(preprocessed.endswith('_preprocessed.'))
    
    def test_base64_encoding(self):
        """Test base64 encoding"""
        base64_str, media_type = self.preprocessor.encode_image_to_base64(self.test_image)
        
        self.assertIsInstance(base64_str, str)
        self.assertGreater(len(base64_str), 0)
        self.assertIn('image/', media_type)


if __name__ == '__main__':
    unittest.main()