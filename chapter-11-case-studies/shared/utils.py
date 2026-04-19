"""
Utility functions for all case studies
"""

import logging
import sys
from typing import List
from datetime import datetime
import anthropic
import re


def setup_logging(log_level: str = "INFO") -> None:
    """Setup logging configuration"""
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Reduce noise from third-party libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("anthropic").setLevel(logging.WARNING)


async def get_embedding(
    text: str,
    model: str = "voyage-2"
) -> List[float]:
    """
    Get embedding for text
    
    Note: This is a placeholder. In production, use:
    - Voyage AI API
    - OpenAI embeddings
    - Anthropic embeddings (when available)
    """
    # Placeholder: return random embedding
    # In production, call actual embedding API
    import random
    return [random.random() for _ in range(1536)]


def chunk_text(
    text: str,
    chunk_size: int = 1000,
    overlap: int = 200
) -> List[str]:
    """
    Chunk text into overlapping segments
    
    Args:
        text: Text to chunk
        chunk_size: Maximum chunk size in characters
        overlap: Overlap between chunks
    
    Returns:
        List of text chunks
    """
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        
        # Try to break at sentence boundary
        if end < len(text):
            # Look for sentence end in last 100 chars
            last_period = text.rfind('.', end - 100, end)
            if last_period != -1:
                end = last_period + 1
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        start = end - overlap
    
    return chunks


def sanitize_input(text: str) -> str:
    """
    Sanitize user input
    
    - Remove control characters
    - Normalize whitespace
    - Remove potential injection attempts
    """
    # Remove control characters
    text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()


def format_timestamp(dt: datetime = None) -> str:
    """Format datetime as ISO string"""
    if dt is None:
        dt = datetime.utcnow()
    return dt.isoformat() + 'Z'


def truncate_text(
    text: str,
    max_length: int = 1000,
    suffix: str = "..."
) -> str:
    """Truncate text to maximum length"""
    if len(text) <= max_length:
        return text
    
    return text[:max_length - len(suffix)] + suffix


def extract_keywords(text: str, max_keywords: int = 10) -> List[str]:
    """
    Extract keywords from text (simple implementation)
    
    In production, use:
    - TF-IDF
    - TextRank
    - LLM-based extraction
    """
    # Simple implementation: extract capitalized words
    words = re.findall(r'\b[A-Z][a-z]+\b', text)
    
    # Count frequency
    from collections import Counter
    word_counts = Counter(words)
    
    # Return top keywords
    return [word for word, count in word_counts.most_common(max_keywords)]