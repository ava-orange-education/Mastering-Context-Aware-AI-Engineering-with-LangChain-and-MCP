"""
Embedding generation and management.
"""

from typing import List, Dict, Any, Optional, Union
import numpy as np
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingResult:
    """Result from embedding generation"""
    text: str
    embedding: np.ndarray
    model: str
    metadata: Dict[str, Any]


class EmbeddingManager:
    """Manage embedding generation with multiple models"""
    
    def __init__(self, 
                 model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 device: str = "cpu"):
        """
        Initialize embedding manager
        
        Args:
            model_name: HuggingFace model name or path
            device: 'cpu' or 'cuda'
        """
        self.model_name = model_name
        self.device = device
        self.model = None
        self.embedding_dim = None
        
        self._load_model()
    
    def _load_model(self):
        """Load embedding model"""
        try:
            from sentence_transformers import SentenceTransformer
            
            self.model = SentenceTransformer(self.model_name, device=self.device)
            
            # Get embedding dimension
            test_embedding = self.model.encode("test")
            self.embedding_dim = len(test_embedding)
            
            logger.info(f"Loaded embedding model: {self.model_name} (dim: {self.embedding_dim})")
        
        except ImportError:
            logger.error("sentence-transformers not installed")
            raise
        except Exception as e:
            logger.error(f"Failed to load model {self.model_name}: {e}")
            raise
    
    def embed(self, 
              texts: Union[str, List[str]],
              batch_size: int = 32,
              show_progress: bool = False) -> Union[EmbeddingResult, List[EmbeddingResult]]:
        """
        Generate embeddings for text(s)
        
        Args:
            texts: Single text or list of texts
            batch_size: Batch size for processing
            show_progress: Show progress bar
            
        Returns:
            EmbeddingResult or list of EmbeddingResults
        """
        if not self.model:
            raise RuntimeError("Model not loaded")
        
        # Handle single text
        single_input = isinstance(texts, str)
        if single_input:
            texts = [texts]
        
        # Generate embeddings
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True
        )
        
        # Create results
        results = []
        for text, embedding in zip(texts, embeddings):
            result = EmbeddingResult(
                text=text,
                embedding=embedding,
                model=self.model_name,
                metadata={
                    'embedding_dim': self.embedding_dim,
                    'model_type': 'sentence-transformer'
                }
            )
            results.append(result)
        
        return results[0] if single_input else results
    
    def embed_batch(self, 
                   texts: List[str],
                   batch_size: int = 32) -> List[EmbeddingResult]:
        """
        Generate embeddings for a batch of texts
        
        Args:
            texts: List of texts
            batch_size: Batch size
            
        Returns:
            List of EmbeddingResults
        """
        return self.embed(texts, batch_size=batch_size, show_progress=True)
    
    def compute_similarity(self, 
                          embedding1: np.ndarray,
                          embedding2: np.ndarray) -> float:
        """
        Compute cosine similarity between embeddings
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            
        Returns:
            Cosine similarity score
        """
        dot_product = np.dot(embedding1, embedding2)
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        return float(dot_product / (norm1 * norm2))
    
    def find_most_similar(self,
                         query_embedding: np.ndarray,
                         candidate_embeddings: List[np.ndarray],
                         top_k: int = 5) -> List[tuple]:
        """
        Find most similar embeddings to query
        
        Args:
            query_embedding: Query embedding
            candidate_embeddings: List of candidate embeddings
            top_k: Number of results to return
            
        Returns:
            List of (index, similarity_score) tuples
        """
        similarities = []
        
        for idx, candidate in enumerate(candidate_embeddings):
            similarity = self.compute_similarity(query_embedding, candidate)
            similarities.append((idx, similarity))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]


class OpenAIEmbeddingManager:
    """Embedding manager for OpenAI models"""
    
    def __init__(self, api_key: str, model: str = "text-embedding-3-small"):
        """
        Initialize OpenAI embedding manager
        
        Args:
            api_key: OpenAI API key
            model: Embedding model name
        """
        self.api_key = api_key
        self.model = model
        self.client = None
        
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize OpenAI client"""
        try:
            from openai import OpenAI
            
            self.client = OpenAI(api_key=self.api_key)
            
            logger.info(f"Initialized OpenAI client with model: {self.model}")
        
        except ImportError:
            logger.error("openai package not installed")
            raise
    
    def embed(self, 
              texts: Union[str, List[str]],
              batch_size: int = 100) -> Union[EmbeddingResult, List[EmbeddingResult]]:
        """
        Generate embeddings using OpenAI API
        
        Args:
            texts: Single text or list of texts
            batch_size: Batch size (max 2048 for OpenAI)
            
        Returns:
            EmbeddingResult or list of EmbeddingResults
        """
        if not self.client:
            raise RuntimeError("Client not initialized")
        
        # Handle single text
        single_input = isinstance(texts, str)
        if single_input:
            texts = [texts]
        
        # Process in batches
        all_results = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            try:
                response = self.client.embeddings.create(
                    input=batch,
                    model=self.model
                )
                
                for j, embedding_data in enumerate(response.data):
                    embedding = np.array(embedding_data.embedding)
                    
                    result = EmbeddingResult(
                        text=batch[j],
                        embedding=embedding,
                        model=self.model,
                        metadata={
                            'embedding_dim': len(embedding),
                            'model_type': 'openai'
                        }
                    )
                    all_results.append(result)
            
            except Exception as e:
                logger.error(f"OpenAI embedding error: {e}")
                raise
        
        return all_results[0] if single_input else all_results


class CohereEmbeddingManager:
    """Embedding manager for Cohere models"""
    
    def __init__(self, api_key: str, model: str = "embed-english-v3.0"):
        """
        Initialize Cohere embedding manager
        
        Args:
            api_key: Cohere API key
            model: Embedding model name
        """
        self.api_key = api_key
        self.model = model
        self.client = None
        
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize Cohere client"""
        try:
            import cohere
            
            self.client = cohere.Client(self.api_key)
            
            logger.info(f"Initialized Cohere client with model: {self.model}")
        
        except ImportError:
            logger.error("cohere package not installed")
            raise
    
    def embed(self,
              texts: Union[str, List[str]],
              input_type: str = "search_document") -> Union[EmbeddingResult, List[EmbeddingResult]]:
        """
        Generate embeddings using Cohere API
        
        Args:
            texts: Single text or list of texts
            input_type: 'search_document', 'search_query', 'classification', or 'clustering'
            
        Returns:
            EmbeddingResult or list of EmbeddingResults
        """
        if not self.client:
            raise RuntimeError("Client not initialized")
        
        # Handle single text
        single_input = isinstance(texts, str)
        if single_input:
            texts = [texts]
        
        try:
            response = self.client.embed(
                texts=texts,
                model=self.model,
                input_type=input_type
            )
            
            results = []
            for text, embedding in zip(texts, response.embeddings):
                embedding_array = np.array(embedding)
                
                result = EmbeddingResult(
                    text=text,
                    embedding=embedding_array,
                    model=self.model,
                    metadata={
                        'embedding_dim': len(embedding_array),
                        'model_type': 'cohere',
                        'input_type': input_type
                    }
                )
                results.append(result)
            
            return results[0] if single_input else results
        
        except Exception as e:
            logger.error(f"Cohere embedding error: {e}")
            raise