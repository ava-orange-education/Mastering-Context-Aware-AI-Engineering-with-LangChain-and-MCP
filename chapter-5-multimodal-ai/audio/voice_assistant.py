"""
Voice assistant with speech-to-text and text-to-speech.
"""

from typing import Optional, Dict, Any
import logging

from .whisper_integration import WhisperIntegration

logger = logging.getLogger(__name__)


class VoiceAssistant:
    """Voice-enabled AI assistant"""
    
    def __init__(self, api_key: str, whisper_model: str = "base"):
        """
        Initialize voice assistant
        
        Args:
            api_key: Anthropic API key
            whisper_model: Whisper model size
        """
        from anthropic import Anthropic
        
        self.client = Anthropic(api_key=api_key)
        self.whisper = WhisperIntegration(whisper_model)
        self.conversation_history = []
    
    def process_voice_input(self, audio_path: str) -> str:
        """
        Process voice input and generate response
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Text response
        """
        # Transcribe audio
        transcription = self.whisper.transcribe(audio_path)
        user_input = transcription['text']
        
        logger.info(f"User said: {user_input}")
        
        # Add to conversation history
        self.conversation_history.append({
            "role": "user",
            "content": user_input
        })
        
        # Generate response
        response = self.client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1024,
            messages=self.conversation_history
        )
        
        assistant_response = response.content[0].text
        
        # Add to history
        self.conversation_history.append({
            "role": "assistant",
            "content": assistant_response
        })
        
        logger.info(f"Assistant: {assistant_response}")
        
        return assistant_response
    
    def text_to_speech(self, text: str, output_path: str):
        """
        Convert text to speech
        
        Args:
            text: Text to convert
            output_path: Path to save audio file
        """
        try:
            from gtts import gTTS
            
            tts = gTTS(text=text, lang='en')
            tts.save(output_path)
            
            logger.info(f"Speech saved to {output_path}")
            
        except ImportError:
            logger.error("gTTS not installed. Install with: pip install gtts")
    
    def voice_conversation(self, audio_path: str, output_audio_path: str) -> str:
        """
        Complete voice conversation: speech-to-text, generate response, text-to-speech
        
        Args:
            audio_path: Input audio
            output_audio_path: Output audio path
            
        Returns:
            Text response
        """
        # Process voice input
        response = self.process_voice_input(audio_path)
        
        # Convert response to speech
        self.text_to_speech(response, output_audio_path)
        
        return response