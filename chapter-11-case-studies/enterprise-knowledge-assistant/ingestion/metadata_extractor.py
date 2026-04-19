"""
Metadata Extractor

Extracts rich metadata from documents for better searchability
"""

from typing import Dict, Any, List, Optional
import logging
import re
from datetime import datetime

logger = logging.getLogger(__name__)


class MetadataExtractor:
    """
    Extracts metadata from parsed documents
    """
    
    def __init__(self):
        # Common document sections
        self.section_patterns = [
            r'^#+\s+(.+)$',  # Markdown headers
            r'^([A-Z][A-Za-z\s]+):',  # Section headers
            r'^\d+\.\s+([A-Z][A-Za-z\s]+)',  # Numbered sections
        ]
        
        # Entity patterns
        self.email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        self.url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
        self.date_pattern = r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b'
    
    def extract(
        self,
        parsed_doc: Any,
        source: Optional[str] = None,
        additional_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Extract metadata from parsed document
        
        Args:
            parsed_doc: ParsedDocument object
            source: Source system (sharepoint, confluence, etc.)
            additional_metadata: Additional metadata to include
        
        Returns:
            Complete metadata dictionary
        """
        
        logger.info(f"Extracting metadata from {parsed_doc.file_type} document")
        
        metadata = parsed_doc.metadata.copy()
        
        # Add source
        if source:
            metadata['source'] = source
        
        # Extract content-based metadata
        content_metadata = self._extract_content_metadata(parsed_doc.content)
        metadata.update(content_metadata)
        
        # Extract entities
        entities = self._extract_entities(parsed_doc.content)
        metadata['entities'] = entities
        
        # Extract topics (simplified - in production use NLP)
        topics = self._extract_topics(parsed_doc.content)
        metadata['topics'] = topics
        
        # Calculate content statistics
        stats = self._calculate_statistics(parsed_doc.content)
        metadata['statistics'] = stats
        
        # Add processing metadata
        metadata['indexed_at'] = datetime.utcnow().isoformat()
        metadata['document_type'] = self._classify_document_type(
            parsed_doc.content,
            parsed_doc.file_type
        )
        
        # Merge additional metadata
        if additional_metadata:
            metadata.update(additional_metadata)
        
        return metadata
    
    def _extract_content_metadata(self, content: str) -> Dict[str, Any]:
        """Extract metadata from content"""
        
        metadata = {}
        
        # Extract title (first significant line)
        lines = [line.strip() for line in content.split('\n') if line.strip()]
        if lines:
            metadata['title'] = lines[0][:200]  # First line, max 200 chars
        
        # Extract sections
        sections = self._extract_sections(content)
        metadata['sections'] = sections
        
        # Extract key phrases (simplified)
        key_phrases = self._extract_key_phrases(content)
        metadata['key_phrases'] = key_phrases
        
        return metadata
    
    def _extract_sections(self, content: str) -> List[str]:
        """Extract document sections"""
        
        sections = []
        
        for pattern in self.section_patterns:
            matches = re.findall(pattern, content, re.MULTILINE)
            sections.extend(matches)
        
        # Deduplicate and limit
        unique_sections = list(dict.fromkeys(sections))[:20]
        
        return unique_sections
    
    def _extract_key_phrases(self, content: str) -> List[str]:
        """Extract key phrases (simplified)"""
        
        # Very simplified - in production use NLP
        # Look for capitalized phrases
        phrases = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b', content)
        
        # Count frequency
        from collections import Counter
        phrase_counts = Counter(phrases)
        
        # Return top phrases
        top_phrases = [phrase for phrase, count in phrase_counts.most_common(10)]
        
        return top_phrases
    
    def _extract_entities(self, content: str) -> Dict[str, List[str]]:
        """Extract named entities"""
        
        entities = {
            "emails": [],
            "urls": [],
            "dates": []
        }
        
        # Extract emails
        emails = re.findall(self.email_pattern, content)
        entities["emails"] = list(set(emails))[:20]
        
        # Extract URLs
        urls = re.findall(self.url_pattern, content)
        entities["urls"] = list(set(urls))[:20]
        
        # Extract dates
        dates = re.findall(self.date_pattern, content)
        entities["dates"] = list(set(dates))[:20]
        
        return entities
    
    def _extract_topics(self, content: str) -> List[str]:
        """Extract topics (simplified keyword extraction)"""
        
        # Very simplified - in production use topic modeling or LLM
        # Extract capitalized words as potential topics
        words = re.findall(r'\b[A-Z][a-z]{2,}\b', content)
        
        # Filter common words
        stop_words = {'The', 'This', 'That', 'These', 'Those', 'When', 'Where', 'What', 'Why', 'How'}
        topics = [word for word in words if word not in stop_words]
        
        # Count and return top topics
        from collections import Counter
        topic_counts = Counter(topics)
        
        return [topic for topic, count in topic_counts.most_common(10)]
    
    def _calculate_statistics(self, content: str) -> Dict[str, Any]:
        """Calculate content statistics"""
        
        words = content.split()
        
        return {
            "character_count": len(content),
            "word_count": len(words),
            "line_count": len(content.split('\n')),
            "average_word_length": sum(len(word) for word in words) / len(words) if words else 0
        }
    
    def _classify_document_type(
        self,
        content: str,
        file_type: str
    ) -> str:
        """Classify document type based on content and format"""
        
        content_lower = content.lower()
        
        # Check for specific document types
        if "meeting notes" in content_lower or "minutes" in content_lower:
            return "meeting_notes"
        elif "proposal" in content_lower and file_type in ['.docx', '.pdf']:
            return "proposal"
        elif "report" in content_lower:
            return "report"
        elif "presentation" in file_type or file_type in ['.pptx', '.ppt']:
            return "presentation"
        elif "spreadsheet" in file_type or file_type in ['.xlsx', '.xls']:
            return "spreadsheet"
        elif "policy" in content_lower or "procedure" in content_lower:
            return "policy_document"
        elif "technical" in content_lower and "specification" in content_lower:
            return "technical_specification"
        else:
            return "general_document"
    
    def extract_author_from_content(self, content: str) -> Optional[str]:
        """Extract author from document content"""
        
        # Look for common author patterns
        author_patterns = [
            r'[Aa]uthor[:\s]+([A-Za-z\s]+)',
            r'[Bb]y[:\s]+([A-Za-z\s]+)',
            r'[Ww]ritten by[:\s]+([A-Za-z\s]+)',
        ]
        
        for pattern in author_patterns:
            match = re.search(pattern, content)
            if match:
                return match.group(1).strip()
        
        return None
    
    def extract_date_from_content(self, content: str) -> Optional[str]:
        """Extract date from document content"""
        
        # Look for date patterns in first 500 characters
        header = content[:500]
        
        date_patterns = [
            r'[Dd]ate[:\s]+(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
            r'(\d{1,2}[/-]\d{1,2}[/-]\d{4})',
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, header)
            if match:
                return match.group(1)
        
        return None