"""
Text cleaning and normalization utilities.
"""

import re
import unicodedata
from typing import List, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class TextCleaner:
    """Comprehensive text cleaning and normalization"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Cleaning options
        self.lowercase = self.config.get('lowercase', False)
        self.remove_urls = self.config.get('remove_urls', True)
        self.remove_emails = self.config.get('remove_emails', True)
        self.remove_phone_numbers = self.config.get('remove_phone_numbers', True)
        self.remove_numbers = self.config.get('remove_numbers', False)
        self.remove_punctuation = self.config.get('remove_punctuation', False)
        self.remove_extra_whitespace = self.config.get('remove_extra_whitespace', True)
        self.normalize_unicode = self.config.get('normalize_unicode', True)
        
        # Regex patterns
        self.url_pattern = re.compile(
            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        )
        self.email_pattern = re.compile(r'\S+@\S+\.\S+')
        self.phone_pattern = re.compile(r'[\+]?[(]?[0-9]{1,4}[)]?[-\s\.]?[(]?[0-9]{1,4}[)]?[-\s\.]?[0-9]{1,9}')
        self.number_pattern = re.compile(r'\b\d+\b')
    
    def clean(self, text: str) -> str:
        """
        Apply all enabled cleaning operations
        
        Args:
            text: Raw text to clean
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Unicode normalization
        if self.normalize_unicode:
            text = self._normalize_unicode(text)
        
        # Remove specific patterns
        if self.remove_urls:
            text = self.url_pattern.sub(' ', text)
        
        if self.remove_emails:
            text = self.email_pattern.sub(' ', text)
        
        if self.remove_phone_numbers:
            text = self.phone_pattern.sub(' ', text)
        
        if self.remove_numbers:
            text = self.number_pattern.sub(' ', text)
        
        # Case normalization
        if self.lowercase:
            text = text.lower()
        
        # Punctuation removal
        if self.remove_punctuation:
            text = re.sub(r'[^\w\s]', ' ', text)
        
        # Whitespace normalization
        if self.remove_extra_whitespace:
            text = ' '.join(text.split())
        
        return text.strip()
    
    def _normalize_unicode(self, text: str) -> str:
        """Normalize unicode characters"""
        # Normalize to NFC form (canonical decomposition followed by canonical composition)
        text = unicodedata.normalize('NFC', text)
        
        # Remove non-printable characters
        text = ''.join(char for char in text if unicodedata.category(char)[0] != 'C')
        
        return text
    
    def remove_html_tags(self, text: str) -> str:
        """Remove HTML tags"""
        clean_text = re.sub(r'<[^>]+>', '', text)
        return clean_text
    
    def remove_markdown(self, text: str) -> str:
        """Remove markdown formatting"""
        # Remove headers
        text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
        
        # Remove bold/italic
        text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
        text = re.sub(r'\*([^*]+)\*', r'\1', text)
        text = re.sub(r'__([^_]+)__', r'\1', text)
        text = re.sub(r'_([^_]+)_', r'\1', text)
        
        # Remove links
        text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
        
        # Remove code blocks
        text = re.sub(r'```[^`]*```', '', text, flags=re.DOTALL)
        text = re.sub(r'`([^`]+)`', r'\1', text)
        
        return text
    
    def extract_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences
        
        Args:
            text: Input text
            
        Returns:
            List of sentences
        """
        # Simple sentence splitting (can be improved with spaCy)
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences
    
    def normalize_whitespace(self, text: str) -> str:
        """Normalize all whitespace to single spaces"""
        return ' '.join(text.split())


class DocumentNormalizer:
    """Normalize documents for consistent processing"""
    
    def __init__(self):
        self.text_cleaner = TextCleaner()
    
    def normalize_document(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize document content based on type
        
        Args:
            content: Document content dictionary
            
        Returns:
            Normalized content
        """
        doc_type = content.get('type', 'text')
        
        if doc_type == 'text':
            return self._normalize_text_document(content)
        elif doc_type == 'pdf':
            return self._normalize_pdf_document(content)
        elif doc_type == 'docx':
            return self._normalize_docx_document(content)
        elif doc_type == 'html':
            return self._normalize_html_document(content)
        elif doc_type == 'json':
            return self._normalize_json_document(content)
        else:
            return content
    
    def _normalize_text_document(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize plain text document"""
        text = content.get('content', '')
        
        # Clean text
        cleaned = self.text_cleaner.clean(text)
        
        return {
            'type': 'text',
            'content': cleaned,
            'normalized': True
        }
    
    def _normalize_pdf_document(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize PDF document"""
        pages = content.get('pages', [])
        
        normalized_pages = []
        for page in pages:
            text = page.get('text', '')
            cleaned = self.text_cleaner.clean(text)
            
            normalized_pages.append({
                'page': page.get('page'),
                'text': cleaned
            })
        
        # Combine all pages
        full_text = '\n\n'.join(p['text'] for p in normalized_pages)
        
        return {
            'type': 'pdf',
            'pages': normalized_pages,
            'full_text': full_text,
            'normalized': True
        }
    
    def _normalize_docx_document(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize DOCX document"""
        paragraphs = content.get('paragraphs', [])
        
        cleaned_paragraphs = [
            self.text_cleaner.clean(p) for p in paragraphs
        ]
        
        full_text = '\n\n'.join(cleaned_paragraphs)
        
        return {
            'type': 'docx',
            'paragraphs': cleaned_paragraphs,
            'full_text': full_text,
            'normalized': True
        }
    
    def _normalize_html_document(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize HTML document"""
        html = content.get('content', '')
        
        # Remove HTML tags
        text = self.text_cleaner.remove_html_tags(html)
        
        # Clean text
        cleaned = self.text_cleaner.clean(text)
        
        return {
            'type': 'html',
            'content': cleaned,
            'normalized': True
        }
    
    def _normalize_json_document(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize JSON document"""
        data = content.get('data', {})
        
        # Extract text from JSON structure
        text_parts = self._extract_text_from_json(data)
        full_text = '\n'.join(text_parts)
        
        # Clean text
        cleaned = self.text_cleaner.clean(full_text)
        
        return {
            'type': 'json',
            'content': cleaned,
            'original_data': data,
            'normalized': True
        }
    
    def _extract_text_from_json(self, data: Any) -> List[str]:
        """Recursively extract text from JSON structure"""
        text_parts = []
        
        if isinstance(data, dict):
            for key, value in data.items():
                text_parts.extend(self._extract_text_from_json(value))
        elif isinstance(data, list):
            for item in data:
                text_parts.extend(self._extract_text_from_json(item))
        elif isinstance(data, str):
            text_parts.append(data)
        elif data is not None:
            text_parts.append(str(data))
        
        return text_parts