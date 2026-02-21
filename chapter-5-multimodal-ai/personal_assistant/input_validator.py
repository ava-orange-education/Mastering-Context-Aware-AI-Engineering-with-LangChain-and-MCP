"""
Input validation for personal assistant.
"""

from typing import Dict, Any, List, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class InputValidator:
    """Validate inputs to the personal assistant"""
    
    SUPPORTED_IMAGE_FORMATS = ['.jpg', '.jpeg', '.png', '.gif', '.webp']
    SUPPORTED_AUDIO_FORMATS = ['.mp3', '.wav', '.m4a', '.flac', '.ogg']
    SUPPORTED_DOCUMENT_FORMATS = ['.pdf', '.docx', '.txt', '.md']
    
    MAX_FILE_SIZE_MB = 20
    MAX_IMAGE_DIMENSION = 4096
    MAX_AUDIO_DURATION_SECONDS = 3600  # 1 hour
    
    @staticmethod
    def validate_file_exists(file_path: str) -> Dict[str, Any]:
        """
        Validate that file exists
        
        Returns:
            Validation result
        """
        path = Path(file_path)
        
        if not path.exists():
            return {
                'valid': False,
                'error': f'File does not exist: {file_path}'
            }
        
        if not path.is_file():
            return {
                'valid': False,
                'error': f'Path is not a file: {file_path}'
            }
        
        return {'valid': True}
    
    @staticmethod
    def validate_file_size(file_path: str, max_size_mb: Optional[float] = None) -> Dict[str, Any]:
        """
        Validate file size
        
        Args:
            file_path: Path to file
            max_size_mb: Maximum size in MB
            
        Returns:
            Validation result
        """
        if max_size_mb is None:
            max_size_mb = InputValidator.MAX_FILE_SIZE_MB
        
        path = Path(file_path)
        size_mb = path.stat().st_size / (1024 * 1024)
        
        if size_mb > max_size_mb:
            return {
                'valid': False,
                'error': f'File too large: {size_mb:.2f}MB (max: {max_size_mb}MB)',
                'size_mb': size_mb
            }
        
        return {'valid': True, 'size_mb': size_mb}
    
    @staticmethod
    def validate_image(image_path: str) -> Dict[str, Any]:
        """
        Validate image file
        
        Args:
            image_path: Path to image
            
        Returns:
            Validation result
        """
        # Check existence
        exists_check = InputValidator.validate_file_exists(image_path)
        if not exists_check['valid']:
            return exists_check
        
        # Check extension
        path = Path(image_path)
        if path.suffix.lower() not in InputValidator.SUPPORTED_IMAGE_FORMATS:
            return {
                'valid': False,
                'error': f'Unsupported image format: {path.suffix}',
                'supported_formats': InputValidator.SUPPORTED_IMAGE_FORMATS
            }
        
        # Check size
        size_check = InputValidator.validate_file_size(image_path)
        if not size_check['valid']:
            return size_check
        
        # Check dimensions
        try:
            from PIL import Image
            img = Image.open(image_path)
            width, height = img.size
            
            if width > InputValidator.MAX_IMAGE_DIMENSION or height > InputValidator.MAX_IMAGE_DIMENSION:
                return {
                    'valid': False,
                    'error': f'Image too large: {width}x{height} (max: {InputValidator.MAX_IMAGE_DIMENSION})',
                    'dimensions': (width, height)
                }
            
            return {
                'valid': True,
                'dimensions': (width, height),
                'format': img.format
            }
            
        except Exception as e:
            return {
                'valid': False,
                'error': f'Could not open image: {str(e)}'
            }
    
    @staticmethod
    def validate_audio(audio_path: str) -> Dict[str, Any]:
        """
        Validate audio file
        
        Args:
            audio_path: Path to audio
            
        Returns:
            Validation result
        """
        # Check existence
        exists_check = InputValidator.validate_file_exists(audio_path)
        if not exists_check['valid']:
            return exists_check
        
        # Check extension
        path = Path(audio_path)
        if path.suffix.lower() not in InputValidator.SUPPORTED_AUDIO_FORMATS:
            return {
                'valid': False,
                'error': f'Unsupported audio format: {path.suffix}',
                'supported_formats': InputValidator.SUPPORTED_AUDIO_FORMATS
            }
        
        # Check size
        size_check = InputValidator.validate_file_size(audio_path)
        if not size_check['valid']:
            return size_check
        
        # Check duration (optional, requires librosa)
        try:
            import librosa
            duration = librosa.get_duration(path=audio_path)
            
            if duration > InputValidator.MAX_AUDIO_DURATION_SECONDS:
                return {
                    'valid': False,
                    'error': f'Audio too long: {duration}s (max: {InputValidator.MAX_AUDIO_DURATION_SECONDS}s)',
                    'duration': duration
                }
            
            return {
                'valid': True,
                'duration': duration
            }
            
        except ImportError:
            # librosa not available, skip duration check
            return {'valid': True}
        except Exception as e:
            return {
                'valid': False,
                'error': f'Could not process audio: {str(e)}'
            }
    
    @staticmethod
    def validate_document(document_path: str) -> Dict[str, Any]:
        """
        Validate document file
        
        Args:
            document_path: Path to document
            
        Returns:
            Validation result
        """
        # Check existence
        exists_check = InputValidator.validate_file_exists(document_path)
        if not exists_check['valid']:
            return exists_check
        
        # Check extension
        path = Path(document_path)
        if path.suffix.lower() not in InputValidator.SUPPORTED_DOCUMENT_FORMATS:
            return {
                'valid': False,
                'error': f'Unsupported document format: {path.suffix}',
                'supported_formats': InputValidator.SUPPORTED_DOCUMENT_FORMATS
            }
        
        # Check size
        size_check = InputValidator.validate_file_size(document_path)
        if not size_check['valid']:
            return size_check
        
        return {'valid': True}
    
    @staticmethod
    def validate_request(request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate complete assistant request
        
        Args:
            request: Request dictionary
            
        Returns:
            Validation result
        """
        errors = []
        
        # Validate required fields
        if 'task' not in request:
            errors.append("Missing required field: 'task'")
        
        # Validate files based on task
        task = request.get('task', '')
        
        if 'image' in task.lower() and 'image_path' in request:
            image_validation = InputValidator.validate_image(request['image_path'])
            if not image_validation['valid']:
                errors.append(image_validation['error'])
        
        if 'audio' in task.lower() and 'audio_path' in request:
            audio_validation = InputValidator.validate_audio(request['audio_path'])
            if not audio_validation['valid']:
                errors.append(audio_validation['error'])
        
        if 'document' in task.lower() and 'document_path' in request:
            doc_validation = InputValidator.validate_document(request['document_path'])
            if not doc_validation['valid']:
                errors.append(doc_validation['error'])
        
        if errors:
            return {
                'valid': False,
                'errors': errors
            }
        
        return {'valid': True}