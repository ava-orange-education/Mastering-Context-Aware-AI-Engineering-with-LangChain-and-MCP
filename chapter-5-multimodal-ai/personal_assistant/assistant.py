"""
Main MultiModalPersonalAssistant implementation.
"""

from typing import Dict, Any, List, Optional
import logging
from .input_validator import InputValidator
from .input_preprocessor import InputPreprocessor
from .cache_manager import MultiModalCache
from .capabilities import AssistantCapabilities

logger = logging.getLogger(__name__)


class MultiModalPersonalAssistant:
    """
    Comprehensive multi-modal personal assistant
    
    Capabilities:
    - Document analysis and Q&A
    - Image understanding and analysis
    - Audio transcription and processing
    - Meeting transcription and summarization
    - Receipt and invoice processing
    - Cross-modal reasoning
    """
    
    def __init__(self, api_key: str, enable_cache: bool = True):
        """
        Initialize personal assistant
        
        Args:
            api_key: Anthropic API key
            enable_cache: Whether to enable response caching
        """
        from mcp_agents import MultiModalOrchestrator
        
        self.orchestrator = MultiModalOrchestrator(api_key)
        self.validator = InputValidator()
        self.preprocessor = InputPreprocessor()
        
        # Initialize cache
        self.cache = MultiModalCache() if enable_cache else None
        
        # Initialize capabilities
        self.capabilities = AssistantCapabilities(self.orchestrator)
        
        logger.info("MultiModalPersonalAssistant initialized")
    
    def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process user request
        
        Args:
            request: Request dictionary with task and parameters
            
        Returns:
            Result dictionary
        """
        # Validate request
        validation = self.validator.validate_request(request)
        if not validation['valid']:
            return {
                'success': False,
                'errors': validation['errors']
            }
        
        # Check cache
        if self.cache:
            cached = self.cache.get(request)
            if cached is not None:
                return {
                    'success': True,
                    'result': cached,
                    'from_cache': True
                }
        
        # Preprocess request
        preprocessed = self.preprocessor.preprocess_request(request)
        
        # Route to appropriate handler
        try:
            result = self._route_request(preprocessed)
            
            # Cache result
            if self.cache:
                self.cache.set(request, result)
            
            return {
                'success': True,
                'result': result,
                'from_cache': False
            }
            
        except Exception as e:
            logger.error(f"Error processing request: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _route_request(self, request: Dict[str, Any]) -> Any:
        """Route request to appropriate capability"""
        
        capability = request.get('capability')
        
        if capability == 'receipt_analysis':
            return self.capabilities.analyze_receipt(request['image_path'])
        
        elif capability == 'meeting_transcription':
            return self.capabilities.transcribe_meeting(request['audio_path'])
        
        elif capability == 'document_verification':
            return self.capabilities.verify_document_authenticity(
                request['document_path'],
                request.get('reference_path')
            )
        
        elif capability == 'cross_modal_reasoning':
            return self.orchestrator.cross_modal_reasoning(
                image_path=request.get('image_path'),
                audio_path=request.get('audio_path'),
                document_path=request.get('document_path'),
                query=request.get('query', '')
            )
        
        else:
            # Default to orchestrator
            return self.orchestrator.process_request(request)
    
    def analyze_document(self, 
                        document_path: str,
                        questions: List[str]) -> Dict[str, Any]:
        """
        Analyze document and answer questions
        
        Args:
            document_path: Path to document
            questions: List of questions
            
        Returns:
            Answers to questions
        """
        return self.process_request({
            'task': 'document_qa',
            'document_path': document_path,
            'questions': questions
        })
    
    def transcribe_audio(self, audio_path: str, language: Optional[str] = None) -> Dict[str, Any]:
        """
        Transcribe audio file
        
        Args:
            audio_path: Path to audio
            language: Optional language code
            
        Returns:
            Transcription result
        """
        return self.process_request({
            'task': 'audio_transcription',
            'audio_path': audio_path,
            'language': language
        })
    
    def analyze_image(self, image_path: str, query: str) -> Dict[str, Any]:
        """
        Analyze image with custom query
        
        Args:
            image_path: Path to image
            query: Question about image
            
        Returns:
            Analysis result
        """
        return self.process_request({
            'task': 'visual_qa',
            'image_path': image_path,
            'question': query
        })
    
    def search_images(self, image_paths: List[str], query: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Search images using text query
        
        Args:
            image_paths: List of image paths
            query: Search query
            top_k: Number of results
            
        Returns:
            Search results
        """
        return self.process_request({
            'task': 'image_search',
            'image_paths': image_paths,
            'query': query,
            'top_k': top_k
        })
    
    def get_stats(self) -> Dict[str, Any]:
        """Get assistant statistics"""
        stats = {
            'orchestrator_logs': len(self.orchestrator.request_log)
        }
        
        if self.cache:
            stats['cache'] = self.cache.get_stats()
        
        return stats