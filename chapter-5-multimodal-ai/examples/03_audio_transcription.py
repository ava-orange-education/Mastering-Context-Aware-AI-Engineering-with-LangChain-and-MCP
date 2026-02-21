"""
Example 3: Audio transcription with Whisper.
"""

import sys
sys.path.append('..')

from audio.whisper_integration import WhisperIntegration


def main():
    print("=== Example 3: Audio Transcription ===\n")
    
    # Initialize Whisper
    print("Loading Whisper model...")
    whisper = WhisperIntegration(model_size="base")
    
    # Example 1: Transcribe English audio
    print("\n1. Transcribing English audio...")
    result = whisper.transcribe("../data/sample_audio/english_speech.mp3")
    print(f"Transcription: {result['text']}")
    print(f"Language: {result['language']}\n")
    
    # Example 2: Transcribe with timestamps
    print("2. Transcribing with timestamps...")
    segments = whisper.transcribe_with_timestamps("../data/sample_audio/meeting.mp3")
    
    for i, segment in enumerate(segments[:3]):  # Show first 3 segments
        print(f"[{segment['start']:.2f}s - {segment['end']:.2f}s]: {segment['text']}")
    print()
    
    # Example 3: Detect language
    print("3. Detecting language...")
    languages = whisper.detect_language("../data/sample_audio/multilingual.mp3")
    
    # Sort by probability
    sorted_langs = sorted(languages.items(), key=lambda x: x[1], reverse=True)
    
    print("Top 5 detected languages:")
    for lang, prob in sorted_langs[:5]:
        print(f"  {lang}: {prob:.2%}")


if __name__ == "__main__":
    main()