"""
Chunking Strategy

Intelligent document chunking for optimal retrieval
"""

from typing import List, Dict, Any, Optional
import logging
import re

logger = logging.getLogger(__name__)


class Chunk:
    """Document chunk"""
    
    def __init__(
        self,
        text: str,
        chunk_id: str,
        metadata: Dict[str, Any],
        start_char: int,
        end_char: int
    ):
        self.text = text
        self.chunk_id = chunk_id
        self.metadata = metadata
        self.start_char = start_char
        self.end_char = end_char
        self.embedding: Optional[List[float]] = None


class ChunkingStrategy:
    """
    Intelligent document chunking strategies
    """
    
    def __init__(
        self,
        chunk_size: int = 1000,
        overlap: int = 200,
        strategy: str = "semantic"
    ):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.strategy = strategy
    
    def chunk_document(
        self,
        content: str,
        metadata: Dict[str, Any],
        document_id: str
    ) -> List[Chunk]:
        """
        Chunk document using selected strategy
        
        Args:
            content: Document content
            metadata: Document metadata
            document_id: Document identifier
        
        Returns:
            List of chunks
        """
        
        logger.info(f"Chunking document {document_id} using {self.strategy} strategy")
        
        if self.strategy == "semantic":
            chunks = self._semantic_chunking(content, metadata, document_id)
        elif self.strategy == "fixed_size":
            chunks = self._fixed_size_chunking(content, metadata, document_id)
        elif self.strategy == "sentence":
            chunks = self._sentence_chunking(content, metadata, document_id)
        elif self.strategy == "section":
            chunks = self._section_chunking(content, metadata, document_id)
        else:
            raise ValueError(f"Unknown chunking strategy: {self.strategy}")
        
        logger.info(f"Created {len(chunks)} chunks for document {document_id}")
        
        return chunks
    
    def _semantic_chunking(
        self,
        content: str,
        metadata: Dict[str, Any],
        document_id: str
    ) -> List[Chunk]:
        """
        Semantic chunking - preserve meaning and context
        Combines section awareness with size constraints
        """
        
        chunks = []
        
        # First, try to identify sections
        sections = self._identify_sections(content)
        
        if sections:
            # Chunk by sections, respecting size limits
            for section_name, section_content in sections:
                section_chunks = self._chunk_section(
                    section_content,
                    metadata,
                    document_id,
                    section_name
                )
                chunks.extend(section_chunks)
        else:
            # Fall back to paragraph-based chunking
            chunks = self._paragraph_chunking(content, metadata, document_id)
        
        return chunks
    
    def _identify_sections(self, content: str) -> List[tuple]:
        """Identify document sections"""
        
        sections = []
        
        # Patterns for section headers
        patterns = [
            r'^#+\s+(.+)$',  # Markdown headers
            r'^([A-Z][A-Za-z\s]+):\s*$',  # Colon headers
            r'^\d+\.\s+([A-Z][A-Za-z\s]+)',  # Numbered sections
        ]
        
        lines = content.split('\n')
        current_section = None
        current_content = []
        
        for line in lines:
            is_header = False
            
            for pattern in patterns:
                match = re.match(pattern, line.strip())
                if match:
                    # Save previous section
                    if current_section and current_content:
                        sections.append((
                            current_section,
                            '\n'.join(current_content)
                        ))
                    
                    current_section = match.group(1)
                    current_content = []
                    is_header = True
                    break
            
            if not is_header and current_section:
                current_content.append(line)
        
        # Save last section
        if current_section and current_content:
            sections.append((current_section, '\n'.join(current_content)))
        
        return sections
    
    def _chunk_section(
        self,
        section_content: str,
        metadata: Dict[str, Any],
        document_id: str,
        section_name: str
    ) -> List[Chunk]:
        """Chunk a document section"""
        
        chunks = []
        
        if len(section_content) <= self.chunk_size:
            # Section fits in one chunk
            chunk = Chunk(
                text=section_content,
                chunk_id=f"{document_id}_section_{len(chunks)}",
                metadata={
                    **metadata,
                    "section": section_name,
                    "chunk_type": "section"
                },
                start_char=0,
                end_char=len(section_content)
            )
            chunks.append(chunk)
        else:
            # Split section into multiple chunks
            paragraphs = section_content.split('\n\n')
            current_chunk = []
            current_length = 0
            
            for para in paragraphs:
                para_length = len(para)
                
                if current_length + para_length <= self.chunk_size:
                    current_chunk.append(para)
                    current_length += para_length
                else:
                    # Save current chunk
                    if current_chunk:
                        chunk_text = '\n\n'.join(current_chunk)
                        chunk = Chunk(
                            text=chunk_text,
                            chunk_id=f"{document_id}_section_{section_name}_{len(chunks)}",
                            metadata={
                                **metadata,
                                "section": section_name,
                                "chunk_type": "section_part"
                            },
                            start_char=0,
                            end_char=len(chunk_text)
                        )
                        chunks.append(chunk)
                    
                    # Start new chunk with overlap
                    if chunks and self.overlap > 0:
                        # Include last paragraph from previous chunk
                        current_chunk = [current_chunk[-1], para]
                        current_length = len(current_chunk[-2]) + para_length
                    else:
                        current_chunk = [para]
                        current_length = para_length
            
            # Save final chunk
            if current_chunk:
                chunk_text = '\n\n'.join(current_chunk)
                chunk = Chunk(
                    text=chunk_text,
                    chunk_id=f"{document_id}_section_{section_name}_{len(chunks)}",
                    metadata={
                        **metadata,
                        "section": section_name,
                        "chunk_type": "section_part"
                    },
                    start_char=0,
                    end_char=len(chunk_text)
                )
                chunks.append(chunk)
        
        return chunks
    
    def _paragraph_chunking(
        self,
        content: str,
        metadata: Dict[str, Any],
        document_id: str
    ) -> List[Chunk]:
        """Chunk by paragraphs"""
        
        chunks = []
        paragraphs = content.split('\n\n')
        
        current_chunk = []
        current_length = 0
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            para_length = len(para)
            
            if current_length + para_length <= self.chunk_size:
                current_chunk.append(para)
                current_length += para_length
            else:
                # Save current chunk
                if current_chunk:
                    chunk_text = '\n\n'.join(current_chunk)
                    chunk = Chunk(
                        text=chunk_text,
                        chunk_id=f"{document_id}_chunk_{len(chunks)}",
                        metadata={**metadata, "chunk_type": "paragraph"},
                        start_char=0,
                        end_char=len(chunk_text)
                    )
                    chunks.append(chunk)
                
                # Start new chunk
                current_chunk = [para]
                current_length = para_length
        
        # Save final chunk
        if current_chunk:
            chunk_text = '\n\n'.join(current_chunk)
            chunk = Chunk(
                text=chunk_text,
                chunk_id=f"{document_id}_chunk_{len(chunks)}",
                metadata={**metadata, "chunk_type": "paragraph"},
                start_char=0,
                end_char=len(chunk_text)
            )
            chunks.append(chunk)
        
        return chunks
    
    def _fixed_size_chunking(
        self,
        content: str,
        metadata: Dict[str, Any],
        document_id: str
    ) -> List[Chunk]:
        """Fixed size chunking with overlap"""
        
        chunks = []
        start = 0
        
        while start < len(content):
            end = start + self.chunk_size
            
            # Try to break at sentence boundary
            if end < len(content):
                # Look for sentence end
                sentence_end = content.rfind('.', start, end)
                if sentence_end > start + (self.chunk_size // 2):
                    end = sentence_end + 1
            
            chunk_text = content[start:end].strip()
            
            if chunk_text:
                chunk = Chunk(
                    text=chunk_text,
                    chunk_id=f"{document_id}_chunk_{len(chunks)}",
                    metadata={**metadata, "chunk_type": "fixed_size"},
                    start_char=start,
                    end_char=end
                )
                chunks.append(chunk)
            
            start = end - self.overlap
        
        return chunks
    
    def _sentence_chunking(
        self,
        content: str,
        metadata: Dict[str, Any],
        document_id: str
    ) -> List[Chunk]:
        """Chunk by sentences"""
        
        # Simple sentence splitting
        sentences = re.split(r'(?<=[.!?])\s+', content)
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            sentence_length = len(sentence)
            
            if current_length + sentence_length <= self.chunk_size:
                current_chunk.append(sentence)
                current_length += sentence_length
            else:
                # Save current chunk
                if current_chunk:
                    chunk_text = ' '.join(current_chunk)
                    chunk = Chunk(
                        text=chunk_text,
                        chunk_id=f"{document_id}_chunk_{len(chunks)}",
                        metadata={**metadata, "chunk_type": "sentence"},
                        start_char=0,
                        end_char=len(chunk_text)
                    )
                    chunks.append(chunk)
                
                # Start new chunk
                current_chunk = [sentence]
                current_length = sentence_length
        
        # Save final chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunk = Chunk(
                text=chunk_text,
                chunk_id=f"{document_id}_chunk_{len(chunks)}",
                metadata={**metadata, "chunk_type": "sentence"},
                start_char=0,
                end_char=len(chunk_text)
            )
            chunks.append(chunk)
        
        return chunks
    
    def _section_chunking(
        self,
        content: str,
        metadata: Dict[str, Any],
        document_id: str
    ) -> List[Chunk]:
        """Chunk by document sections only"""
        
        sections = self._identify_sections(content)
        chunks = []
        
        for section_name, section_content in sections:
            chunk = Chunk(
                text=section_content,
                chunk_id=f"{document_id}_section_{len(chunks)}",
                metadata={
                    **metadata,
                    "section": section_name,
                    "chunk_type": "full_section"
                },
                start_char=0,
                end_char=len(section_content)
            )
            chunks.append(chunk)
        
        return chunks if chunks else self._fixed_size_chunking(content, metadata, document_id)