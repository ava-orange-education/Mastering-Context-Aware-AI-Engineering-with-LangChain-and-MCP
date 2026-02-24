"""
Document chunking strategies for optimal retrieval.
"""

from typing import List, Dict, Any, Optional
import re
import logging

logger = logging.getLogger(__name__)


class Chunk:
    """Represents a document chunk"""
    
    def __init__(self, text: str, metadata: Dict[str, Any], chunk_id: str):
        self.text = text
        self.metadata = metadata
        self.chunk_id = chunk_id
        self.token_count = self._estimate_tokens(text)
    
    def _estimate_tokens(self, text: str) -> int:
        """Rough token count estimation (words * 1.3)"""
        return int(len(text.split()) * 1.3)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'chunk_id': self.chunk_id,
            'text': self.text,
            'metadata': self.metadata,
            'token_count': self.token_count
        }


class ChunkingStrategy:
    """Base class for chunking strategies"""
    
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def chunk(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Chunk]:
        """
        Chunk text into smaller pieces
        
        Args:
            text: Text to chunk
            metadata: Optional metadata to attach to chunks
            
        Returns:
            List of Chunk objects
        """
        raise NotImplementedError


class FixedSizeChunking(ChunkingStrategy):
    """Fixed-size chunking with overlap"""
    
    def chunk(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Chunk]:
        """Chunk text into fixed-size pieces"""
        metadata = metadata or {}
        
        words = text.split()
        chunks = []
        
        start = 0
        chunk_num = 0
        
        while start < len(words):
            end = start + self.chunk_size
            chunk_words = words[start:end]
            chunk_text = ' '.join(chunk_words)
            
            chunk = Chunk(
                text=chunk_text,
                metadata={
                    **metadata,
                    'chunk_index': chunk_num,
                    'start_word': start,
                    'end_word': min(end, len(words)),
                    'chunking_strategy': 'fixed_size'
                },
                chunk_id=f"chunk_{chunk_num}"
            )
            
            chunks.append(chunk)
            
            # Move window with overlap
            start += (self.chunk_size - self.chunk_overlap)
            chunk_num += 1
        
        return chunks


class SentenceChunking(ChunkingStrategy):
    """Chunk by sentences, respecting chunk size limits"""
    
    def chunk(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Chunk]:
        """Chunk text by sentences"""
        metadata = metadata or {}
        
        # Split into sentences
        sentences = self._split_sentences(text)
        
        chunks = []
        current_chunk = []
        current_size = 0
        chunk_num = 0
        
        for sentence in sentences:
            sentence_size = len(sentence.split())
            
            # If single sentence exceeds chunk size, split it
            if sentence_size > self.chunk_size:
                if current_chunk:
                    # Save current chunk
                    chunk_text = ' '.join(current_chunk)
                    chunks.append(self._create_chunk(chunk_text, metadata, chunk_num))
                    chunk_num += 1
                    current_chunk = []
                    current_size = 0
                
                # Split long sentence
                long_sentence_chunks = self._split_long_sentence(sentence, metadata, chunk_num)
                chunks.extend(long_sentence_chunks)
                chunk_num += len(long_sentence_chunks)
                
            elif current_size + sentence_size > self.chunk_size:
                # Save current chunk
                chunk_text = ' '.join(current_chunk)
                chunks.append(self._create_chunk(chunk_text, metadata, chunk_num))
                chunk_num += 1
                
                # Start new chunk with overlap
                if self.chunk_overlap > 0 and len(current_chunk) > 1:
                    overlap_sentences = current_chunk[-1:]
                    current_chunk = overlap_sentences + [sentence]
                    current_size = sum(len(s.split()) for s in current_chunk)
                else:
                    current_chunk = [sentence]
                    current_size = sentence_size
            else:
                current_chunk.append(sentence)
                current_size += sentence_size
        
        # Save final chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append(self._create_chunk(chunk_text, metadata, chunk_num))
        
        return chunks
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        # Simple sentence splitting
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _split_long_sentence(self, sentence: str, metadata: Dict[str, Any], base_chunk_num: int) -> List[Chunk]:
        """Split a sentence that's too long"""
        words = sentence.split()
        chunks = []
        
        start = 0
        sub_chunk_num = 0
        
        while start < len(words):
            end = start + self.chunk_size
            chunk_words = words[start:end]
            chunk_text = ' '.join(chunk_words)
            
            chunk = Chunk(
                text=chunk_text,
                metadata={
                    **metadata,
                    'chunk_index': base_chunk_num + sub_chunk_num,
                    'is_split_sentence': True,
                    'chunking_strategy': 'sentence'
                },
                chunk_id=f"chunk_{base_chunk_num}_{sub_chunk_num}"
            )
            
            chunks.append(chunk)
            start = end
            sub_chunk_num += 1
        
        return chunks
    
    def _create_chunk(self, text: str, metadata: Dict[str, Any], chunk_num: int) -> Chunk:
        """Create a chunk object"""
        return Chunk(
            text=text,
            metadata={
                **metadata,
                'chunk_index': chunk_num,
                'chunking_strategy': 'sentence'
            },
            chunk_id=f"chunk_{chunk_num}"
        )


class SemanticChunking(ChunkingStrategy):
    """Chunk by semantic similarity using embeddings"""
    
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50, similarity_threshold: float = 0.7):
        super().__init__(chunk_size, chunk_overlap)
        self.similarity_threshold = similarity_threshold
        self.embedding_model = None
    
    def _load_embedding_model(self):
        """Load sentence embedding model"""
        if self.embedding_model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            except ImportError:
                logger.error("sentence-transformers not installed")
                raise
    
    def chunk(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Chunk]:
        """Chunk text by semantic similarity"""
        metadata = metadata or {}
        
        self._load_embedding_model()
        
        # Split into sentences
        sentences = self._split_sentences(text)
        
        if not sentences:
            return []
        
        # Get sentence embeddings
        embeddings = self.embedding_model.encode(sentences)
        
        # Group sentences by similarity
        chunks = []
        current_group = [sentences[0]]
        current_size = len(sentences[0].split())
        chunk_num = 0
        
        for i in range(1, len(sentences)):
            # Calculate similarity with previous sentence
            similarity = self._cosine_similarity(embeddings[i-1], embeddings[i])
            
            sentence_size = len(sentences[i].split())
            
            # Check if we should start a new chunk
            if (similarity < self.similarity_threshold or 
                current_size + sentence_size > self.chunk_size):
                
                # Save current chunk
                chunk_text = ' '.join(current_group)
                chunks.append(self._create_chunk(chunk_text, metadata, chunk_num))
                chunk_num += 1
                
                # Start new chunk
                current_group = [sentences[i]]
                current_size = sentence_size
            else:
                current_group.append(sentences[i])
                current_size += sentence_size
        
        # Save final chunk
        if current_group:
            chunk_text = ' '.join(current_group)
            chunks.append(self._create_chunk(chunk_text, metadata, chunk_num))
        
        return chunks
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _cosine_similarity(self, vec1, vec2):
        """Calculate cosine similarity between two vectors"""
        import numpy as np
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    
    def _create_chunk(self, text: str, metadata: Dict[str, Any], chunk_num: int) -> Chunk:
        """Create a chunk object"""
        return Chunk(
            text=text,
            metadata={
                **metadata,
                'chunk_index': chunk_num,
                'chunking_strategy': 'semantic'
            },
            chunk_id=f"chunk_{chunk_num}"
        )


class HierarchicalChunking(ChunkingStrategy):
    """Hierarchical chunking with parent-child relationships"""
    
    def __init__(self, parent_size: int = 2048, child_size: int = 512, child_overlap: int = 50):
        self.parent_size = parent_size
        self.child_size = child_size
        self.child_overlap = child_overlap
    
    def chunk(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Chunk]:
        """Create hierarchical chunks"""
        metadata = metadata or {}
        
        # Create parent chunks
        parent_chunker = FixedSizeChunking(
            chunk_size=self.parent_size,
            chunk_overlap=0
        )
        parent_chunks = parent_chunker.chunk(text, metadata)
        
        # Create child chunks for each parent
        all_chunks = []
        
        for parent_idx, parent_chunk in enumerate(parent_chunks):
            parent_id = f"parent_{parent_idx}"
            
            # Add parent chunk
            parent_chunk.metadata['is_parent'] = True
            parent_chunk.metadata['parent_id'] = parent_id
            parent_chunk.chunk_id = parent_id
            all_chunks.append(parent_chunk)
            
            # Create child chunks
            child_chunker = FixedSizeChunking(
                chunk_size=self.child_size,
                chunk_overlap=self.child_overlap
            )
            child_chunks = child_chunker.chunk(parent_chunk.text, metadata)
            
            for child_idx, child_chunk in enumerate(child_chunks):
                child_chunk.metadata['is_parent'] = False
                child_chunk.metadata['parent_id'] = parent_id
                child_chunk.metadata['child_index'] = child_idx
                child_chunk.chunk_id = f"{parent_id}_child_{child_idx}"
                all_chunks.append(child_chunk)
        
        return all_chunks


class DocumentChunker:
    """Main document chunker with strategy selection"""
    
    def __init__(self, strategy: str = 'sentence', **kwargs):
        """
        Initialize chunker with specified strategy
        
        Args:
            strategy: 'fixed', 'sentence', 'semantic', or 'hierarchical'
            **kwargs: Strategy-specific parameters
        """
        self.strategy_name = strategy
        
        if strategy == 'fixed':
            self.strategy = FixedSizeChunking(**kwargs)
        elif strategy == 'sentence':
            self.strategy = SentenceChunking(**kwargs)
        elif strategy == 'semantic':
            self.strategy = SemanticChunking(**kwargs)
        elif strategy == 'hierarchical':
            self.strategy = HierarchicalChunking(**kwargs)
        else:
            raise ValueError(f"Unknown chunking strategy: {strategy}")
    
    def chunk_document(self, document: Dict[str, Any]) -> List[Chunk]:
        """
        Chunk a document
        
        Args:
            document: Document with 'content' and 'metadata' keys
            
        Returns:
            List of Chunk objects
        """
        content = document.get('content', '')
        metadata = document.get('metadata', {})
        
        if not content:
            return []
        
        # Extract text based on content type
        if isinstance(content, str):
            text = content
        elif isinstance(content, dict):
            text = content.get('full_text', content.get('content', ''))
        else:
            text = str(content)
        
        # Chunk text
        chunks = self.strategy.chunk(text, metadata)
        
        logger.info(f"Chunked document into {len(chunks)} chunks using {self.strategy_name} strategy")
        
        return chunks