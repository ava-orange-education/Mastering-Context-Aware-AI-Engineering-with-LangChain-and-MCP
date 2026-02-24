"""
Metadata extraction from documents.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
import re
import logging

logger = logging.getLogger(__name__)


class MetadataExtractor:
    """Extract metadata from document content"""
    
    def __init__(self):
        self.extractors = {
            'dates': self._extract_dates,
            'emails': self._extract_emails,
            'urls': self._extract_urls,
            'phone_numbers': self._extract_phone_numbers,
            'keywords': self._extract_keywords,
            'entities': self._extract_entities,
            'language': self._detect_language,
            'document_stats': self._compute_document_stats
        }
    
    def extract(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Extract all metadata from text
        
        Args:
            text: Document text
            metadata: Existing metadata to enhance
            
        Returns:
            Enhanced metadata dictionary
        """
        metadata = metadata or {}
        
        for extractor_name, extractor_func in self.extractors.items():
            try:
                result = extractor_func(text)
                if result:
                    metadata[extractor_name] = result
            except Exception as e:
                logger.error(f"Error in {extractor_name} extraction: {e}")
        
        return metadata
    
    def _extract_dates(self, text: str) -> List[str]:
        """Extract date patterns from text"""
        date_patterns = [
            r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}',  # MM/DD/YYYY or DD-MM-YYYY
            r'\d{4}[/-]\d{1,2}[/-]\d{1,2}',    # YYYY-MM-DD
            r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4}\b'  # Month DD, YYYY
        ]
        
        dates = []
        for pattern in date_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            dates.extend(matches)
        
        return list(set(dates))
    
    def _extract_emails(self, text: str) -> List[str]:
        """Extract email addresses"""
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = re.findall(email_pattern, text)
        return list(set(emails))
    
    def _extract_urls(self, text: str) -> List[str]:
        """Extract URLs"""
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        urls = re.findall(url_pattern, text)
        return list(set(urls))
    
    def _extract_phone_numbers(self, text: str) -> List[str]:
        """Extract phone numbers"""
        phone_pattern = r'[\+]?[(]?[0-9]{1,4}[)]?[-\s\.]?[(]?[0-9]{1,4}[)]?[-\s\.]?[0-9]{1,9}'
        phones = re.findall(phone_pattern, text)
        return list(set(phones))
    
    def _extract_keywords(self, text: str, top_n: int = 10) -> List[str]:
        """Extract top keywords using simple frequency analysis"""
        # Remove common stop words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'be',
            'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
            'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this',
            'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they'
        }
        
        # Tokenize and count
        words = re.findall(r'\b[a-z]{3,}\b', text.lower())
        word_freq = {}
        
        for word in words:
            if word not in stop_words:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Get top N
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        keywords = [word for word, freq in sorted_words[:top_n]]
        
        return keywords
    
    def _extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract named entities (requires spaCy)"""
        try:
            import spacy
            
            # Load model (cache it in production)
            nlp = spacy.load("en_core_web_sm")
            
            doc = nlp(text[:100000])  # Limit for performance
            
            entities = {
                'persons': [],
                'organizations': [],
                'locations': [],
                'dates': [],
                'money': []
            }
            
            for ent in doc.ents:
                if ent.label_ == 'PERSON':
                    entities['persons'].append(ent.text)
                elif ent.label_ == 'ORG':
                    entities['organizations'].append(ent.text)
                elif ent.label_ in ['GPE', 'LOC']:
                    entities['locations'].append(ent.text)
                elif ent.label_ == 'DATE':
                    entities['dates'].append(ent.text)
                elif ent.label_ == 'MONEY':
                    entities['money'].append(ent.text)
            
            # Remove duplicates
            for key in entities:
                entities[key] = list(set(entities[key]))
            
            return entities
        
        except ImportError:
            logger.warning("spaCy not installed. Entity extraction unavailable.")
            return {}
        except Exception as e:
            logger.error(f"Entity extraction error: {e}")
            return {}
    
    def _detect_language(self, text: str) -> Optional[str]:
        """Detect document language"""
        try:
            from langdetect import detect
            
            # Use first 1000 characters for detection
            sample = text[:1000]
            language = detect(sample)
            
            return language
        
        except ImportError:
            logger.warning("langdetect not installed. Language detection unavailable.")
            return None
        except Exception as e:
            logger.error(f"Language detection error: {e}")
            return None
    
    def _compute_document_stats(self, text: str) -> Dict[str, Any]:
        """Compute basic document statistics"""
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        
        return {
            'char_count': len(text),
            'word_count': len(words),
            'sentence_count': len([s for s in sentences if s.strip()]),
            'avg_word_length': sum(len(w) for w in words) / len(words) if words else 0,
            'avg_sentence_length': len(words) / len(sentences) if sentences else 0
        }


class StructuredMetadataExtractor:
    """Extract structured metadata from specific document types"""
    
    def extract_invoice_metadata(self, text: str) -> Dict[str, Any]:
        """Extract metadata from invoice documents"""
        metadata = {}
        
        # Invoice number
        invoice_pattern = r'Invoice\s*#?\s*:?\s*([A-Z0-9-]+)'
        match = re.search(invoice_pattern, text, re.IGNORECASE)
        if match:
            metadata['invoice_number'] = match.group(1)
        
        # Amount patterns
        amount_pattern = r'\$\s*([0-9,]+\.?[0-9]*)'
        amounts = re.findall(amount_pattern, text)
        if amounts:
            metadata['amounts'] = [amt.replace(',', '') for amt in amounts]
        
        # Date patterns
        date_pattern = r'Date:\s*([0-9]{1,2}/[0-9]{1,2}/[0-9]{2,4})'
        match = re.search(date_pattern, text, re.IGNORECASE)
        if match:
            metadata['invoice_date'] = match.group(1)
        
        return metadata
    
    def extract_contract_metadata(self, text: str) -> Dict[str, Any]:
        """Extract metadata from contract documents"""
        metadata = {}
        
        # Parties
        party_pattern = r'between\s+([^,]+)\s+and\s+([^,\.]+)'
        match = re.search(party_pattern, text, re.IGNORECASE)
        if match:
            metadata['party_1'] = match.group(1).strip()
            metadata['party_2'] = match.group(2).strip()
        
        # Effective date
        effective_pattern = r'Effective\s+Date:\s*([0-9]{1,2}/[0-9]{1,2}/[0-9]{2,4})'
        match = re.search(effective_pattern, text, re.IGNORECASE)
        if match:
            metadata['effective_date'] = match.group(1)
        
        # Term/Duration
        term_pattern = r'for\s+a\s+term\s+of\s+([0-9]+)\s+(day|week|month|year)s?'
        match = re.search(term_pattern, text, re.IGNORECASE)
        if match:
            metadata['term'] = f"{match.group(1)} {match.group(2)}s"
        
        return metadata
    
    def extract_resume_metadata(self, text: str) -> Dict[str, Any]:
        """Extract metadata from resume documents"""
        metadata = {}
        
        # Skills section
        skills_section = re.search(
            r'Skills:?\s*([^\n]+(?:\n[^\n]+)*?)(?=\n\n|\n[A-Z]|$)',
            text,
            re.IGNORECASE
        )
        if skills_section:
            skills_text = skills_section.group(1)
            # Extract individual skills
            skills = re.findall(r'\b[A-Z][a-z]+\b|\b[A-Z]+\b', skills_text)
            metadata['skills'] = list(set(skills))
        
        # Education
        education_pattern = r'(Bachelor|Master|PhD|B\.S\.|M\.S\.|Ph\.D\.)[^\n]+'
        education = re.findall(education_pattern, text, re.IGNORECASE)
        if education:
            metadata['education'] = education
        
        # Years of experience (heuristic)
        exp_pattern = r'([0-9]+)\+?\s+years?\s+(?:of\s+)?experience'
        match = re.search(exp_pattern, text, re.IGNORECASE)
        if match:
            metadata['years_experience'] = match.group(1)
        
        return metadata