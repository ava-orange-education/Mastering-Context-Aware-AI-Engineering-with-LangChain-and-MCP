"""
Tests for personal assistant.
"""

import unittest
import sys
sys.path.append('..')

from personal_assistant.input_validator import InputValidator
from personal_assistant.cache_manager import MultiModalCache


class TestPersonalAssistant(unittest.TestCase):
    """Test personal assistant components"""
    
    def test_file_validation(self):
        """Test file validation"""
        # Test valid file
        result = InputValidator.validate_file_exists('../data/test_data/test_image.jpg')
        self.assertTrue(result['valid'])
        
        # Test invalid file
        result = InputValidator.validate_file_exists('nonexistent.jpg')
        self.assertFalse(result['valid'])
    
    def test_image_validation(self):
        """Test image validation"""
        result = InputValidator.validate_image('../data/test_data/test_image.jpg')
        
        if result['valid']:
            self.assertIn('dimensions', result)
            self.assertIsInstance(result['dimensions'], tuple)
    
    def test_cache_operations(self):
        """Test cache manager"""
        cache = MultiModalCache(cache_dir='./test_cache')
        
        request = {
            'task': 'test_task',
            'param': 'value'
        }
        
        response = {'result': 'test_result'}
        
        # Test set
        cache.set(request, response)
        
        # Test get
        cached = cache.get(request)
        self.assertIsNotNone(cached)
        self.assertEqual(cached, response)
        
        # Test stats
        stats = cache.get_stats()
        self.assertIn('hits', stats)
        self.assertIn('misses', stats)
        
        # Cleanup
        cache.clear()


if __name__ == '__main__':
    unittest.main()