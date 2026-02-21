"""
Specialized audio agent for MCP system.
"""

from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)


class AudioAgent:
    """Agent specialized in audio tasks"""
    
    def __init__(self, api_key: str):
        from audio import WhisperIntegration, VoiceAssistant
        
        self.whisper = WhisperIntegration()
        self.voice_assistant = VoiceAssistant(api_key)
        
        self.capabilities = [
            'audio_transcription',
            'language_detection',
            'voice_conversation',
            'audio_translation'
        ]
    
    def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle audio-related request
        
        Args:
            request: Request dictionary with 'task' and parameters
            
        Returns:
            Result dictionary
        """
        task = request.get('task')
        
        if task == 'audio_transcription':
            return self._transcribe_audio(request)
        elif task == 'language_detection':
            return self._detect_language(request)
        elif task == 'voice_conversation':
            return self._voice_conversation(request)
        elif task == 'audio_translation':
            return self._translate_audio(request)
        else:
            return {'error': f'Unknown task: {task}'}
    
    def _transcribe_audio(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Transcribe audio file"""
        audio_path = request['audio_path']
        language = request.get('language')
        
        result = self.whisper.transcribe(audio_path, language=language)
        
        return {
            'task': 'audio_transcription',
            'audio_path': audio_path,
            'transcription': result
        }
    
    def _detect_language(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Detect language in audio"""
        audio_path = request['audio_path']
        
        languages = self.whisper.detect_language(audio_path)
        
        return {
            'task': 'language_detection',
            'audio_path': audio_path,
            'detected_languages': languages
        }
    
    def _voice_conversation(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Process voice conversation"""
        audio_path = request['audio_path']
        output_path = request.get('output_path', 'response.mp3')
        
        response = self.voice_assistant.voice_conversation(audio_path, output_path)
        
        return {
            'task': 'voice_conversation',
            'audio_path': audio_path,
            'response_text': response,
            'response_audio': output_path
        }
    
    def _translate_audio(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Translate audio to English"""
        audio_path = request['audio_path']
        
        result = self.whisper.transcribe(audio_path, task='translate')
        
        return {
            'task': 'audio_translation',
            'audio_path': audio_path,
            'translation': result
        }