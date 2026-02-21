"""
LLM integration module for multimodal AI.
"""

import warnings

from .claude_multimodal import ClaudeMultimodal
from .gpt4_vision import GPT4Vision

try:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        from .gemini_integration import GeminiMultimodal
except ImportError:
    GeminiMultimodal = None  # pip install google-generativeai to enable (legacy; prefer google-genai)

__all__ = [
    'ClaudeMultimodal',
    'GPT4Vision',
    'GeminiMultimodal',
]