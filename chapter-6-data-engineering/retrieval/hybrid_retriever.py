"""
Hybrid retrieval combining vector search with keyword/BM25 search.
"""

from typing import List, Dict, Any, Optional
import numpy as np
import logging

logger = logging.getLogger(__name__)


class HybridRetriever:
    """Hybrid search combining dense and sparse retrieval"""
    
    def __init__(self, vector_store, bm25_index=None, alpha: float = 0.5):
        """
        Initialize hybrid retriever
        
        Args:
            vector_store: Vector store for dense retrieval
            bm25_index: BM25 index for sparse retrieval (optional)
            alpha: Weight for dense retrieval (1-alpha for sparse)
        """
        self.vector_store = vector_store
        self.bm25_index = bm25_index
        self.alpha = alpha
    
    def search(self, query: str, query_vector: np.ndarray, 
               top_k: int = 10, 
               filter: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """
        Perform hybrid search
        
        Args:
            query: Text query for keyword search
            query_vector: Embedding vector for semantic search
            top_k: Number of results to return
            filter: Optional metadata filter
            
        Returns:
            List of search results with combined scores
        """
        # Dense retrieval (vector search)
        dense_results = self.vector_store.search(
            query_vector=query_vector,
            top_k=top_k * 2,  # Retrieve more for fusion
            filter=filter
        )
        
        # Sparse retrieval (keyword search)
        if self.bm25_index:
            sparse_results = self.bm25_index.search(query, top_k=top_k * 2)
        else:
            sparse_results = []
        
        # Combine results
        combined_results = self._fuse_results(dense_results, sparse_results, top_k)
        
        return combined_results
    
    def _fuse_results(self, dense_results: List[Dict], sparse_results: List[Dict], top_k: int) -> List[Dict]:
        """
        Fuse dense and sparse results using Reciprocal Rank Fusion
        
        Args:
            dense_results: Results from vector search
            sparse_results: Results from keyword search
            top_k: Number of final results
            
        Returns:
            Fused and ranked results
        """
        # Normalize scores
        dense_scores = self._normalize_scores([r.get('distance', r.get('score', 0)) for r in dense_results])
        sparse_scores = self._normalize_scores([r.get('score', 0) for r in sparse_results])
        
        # Create score dictionary
        all_scores = {}
        
        # Add dense scores
        for i, result in enumerate(dense_results):
            doc_id = result.get('id', str(i))
            all_scores[doc_id] = {
                'dense_score': dense_scores[i] * self.alpha,
                'sparse_score': 0,
                'result': result
            }
        
        # Add sparse scores
        for i, result in enumerate(sparse_results):
            doc_id = result.get('id', str(i))
            
            if doc_id in all_scores:
                all_scores[doc_id]['sparse_score'] = sparse_scores[i] * (1 - self.alpha)
            else:
                all_scores[doc_id] = {
                    'dense_score': 0,
                    'sparse_score': sparse_scores[i] * (1 - self.alpha),
                    'result': result
                }
        
        # Calculate combined scores
        for doc_id in all_scores:
            all_scores[doc_id]['combined_score'] = (
                all_scores[doc_id]['dense_score'] + 
                all_scores[doc_id]['sparse_score']
            )
        
        # Sort by combined score
        sorted_results = sorted(
            all_scores.items(),
            key=lambda x: x[1]['combined_score'],
            reverse=True
        )
        
        # Format final results
        final_results = []
        for doc_id, scores in sorted_results[:top_k]:
            result = scores['result'].copy()
            result['combined_score'] = scores['combined_score']
            result['dense_score'] = scores['dense_score']
            result['sparse_score'] = scores['sparse_score']
            final_results.append(result)
        
        return final_results
    
    def _normalize_scores(self, scores: List[float]) -> List[float]:
        """Min-max normalize scores to [0, 1]"""
        if not scores:
            return []
        
        min_score = min(scores)
        max_score = max(scores)
        
        if max_score == min_score:
            return [1.0] * len(scores)
        
        return [(s - min_score) / (max_score - min_score) for s in scores]


class BM25Index:
    """BM25 sparse retrieval index"""
    
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        """
        Initialize BM25 index
        
        Args:
            k1: Term frequency saturation parameter
            b: Length normalization parameter
        """
        try:
            from rank_bm25 import BM25Okapi
            self.BM25Okapi = BM25Okapi
        except ImportError:
            logger.error("rank-bm25 not installed. Install with: pip install rank-bm25")
            raise
        
        self.k1 = k1
        self.b = b
        self.bm25 = None
        self.documents = []
        self.metadata = []
    
    def build_index(self, documents: List[str], metadata: List[Dict[str, Any]]):
        """
        Build BM25 index from documents
        
        Args:
            documents: List of document texts
            metadata: List of document metadata
        """
        # Tokenize documents
        tokenized_docs = [doc.lower().split() for doc in documents]
        
        # Create BM25 index
        self.bm25 = self.BM25Okapi(tokenized_docs)
        self.documents = documents
        self.metadata = metadata
        
        logger.info(f"Built BM25 index with {len(documents)} documents")
    
    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Search BM25 index
        
        Args:
            query: Query text
            top_k: Number of results
            
        Returns:
            List of search results
        """
        if not self.bm25:
            raise RuntimeError("BM25 index not built. Call build_index first.")
        
        # Tokenize query
        tokenized_query = query.lower().split()
        
        # Get scores
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top k indices
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        # Format results
        results = []
        for idx in top_indices:
            results.append({
                'id': str(idx),
                'document': self.documents[idx],
                'metadata': self.metadata[idx],
                'score': float(scores[idx])
            })
        
        return results