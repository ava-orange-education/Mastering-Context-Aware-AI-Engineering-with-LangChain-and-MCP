"""
Retrieval Quality Evaluator

Evaluates quality of document retrieval
"""

from typing import List, Dict, Any, Set
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


class RetrievalQualityEvaluator:
    """
    Evaluator for retrieval quality metrics
    """
    
    def __init__(self):
        pass
    
    def evaluate(
        self,
        queries: List[str],
        retrieved_docs: List[List[str]],
        relevant_docs: List[List[str]]
    ) -> Dict[str, float]:
        """
        Evaluate retrieval quality
        
        Args:
            queries: List of queries
            retrieved_docs: Retrieved document IDs per query
            relevant_docs: Ground truth relevant document IDs per query
        
        Returns:
            Dictionary of metrics
        """
        
        if len(queries) != len(retrieved_docs) != len(relevant_docs):
            raise ValueError("Queries, retrieved, and relevant docs must have same length")
        
        metrics = {
            "precision_at_1": self.precision_at_k(retrieved_docs, relevant_docs, k=1),
            "precision_at_5": self.precision_at_k(retrieved_docs, relevant_docs, k=5),
            "precision_at_10": self.precision_at_k(retrieved_docs, relevant_docs, k=10),
            "recall_at_1": self.recall_at_k(retrieved_docs, relevant_docs, k=1),
            "recall_at_5": self.recall_at_k(retrieved_docs, relevant_docs, k=5),
            "recall_at_10": self.recall_at_k(retrieved_docs, relevant_docs, k=10),
            "f1_at_10": self.f1_at_k(retrieved_docs, relevant_docs, k=10),
            "mrr": self.mean_reciprocal_rank(retrieved_docs, relevant_docs),
            "map": self.mean_average_precision(retrieved_docs, relevant_docs),
            "ndcg_at_10": self.ndcg_at_k(retrieved_docs, relevant_docs, k=10)
        }
        
        logger.info(f"Evaluated {len(queries)} queries")
        logger.info(f"Precision@10: {metrics['precision_at_10']:.3f}")
        logger.info(f"Recall@10: {metrics['recall_at_10']:.3f}")
        logger.info(f"NDCG@10: {metrics['ndcg_at_10']:.3f}")
        
        return metrics
    
    def precision_at_k(
        self,
        retrieved: List[List[str]],
        relevant: List[List[str]],
        k: int
    ) -> float:
        """Calculate Precision@K"""
        
        precisions = []
        
        for ret, rel in zip(retrieved, relevant):
            ret_k = set(ret[:k])
            rel_set = set(rel)
            
            if len(ret_k) == 0:
                precisions.append(0.0)
            else:
                precision = len(ret_k & rel_set) / len(ret_k)
                precisions.append(precision)
        
        return sum(precisions) / len(precisions) if precisions else 0.0
    
    def recall_at_k(
        self,
        retrieved: List[List[str]],
        relevant: List[List[str]],
        k: int
    ) -> float:
        """Calculate Recall@K"""
        
        recalls = []
        
        for ret, rel in zip(retrieved, relevant):
            ret_k = set(ret[:k])
            rel_set = set(rel)
            
            if len(rel_set) == 0:
                recalls.append(0.0)
            else:
                recall = len(ret_k & rel_set) / len(rel_set)
                recalls.append(recall)
        
        return sum(recalls) / len(recalls) if recalls else 0.0
    
    def f1_at_k(
        self,
        retrieved: List[List[str]],
        relevant: List[List[str]],
        k: int
    ) -> float:
        """Calculate F1@K"""
        
        precision = self.precision_at_k(retrieved, relevant, k)
        recall = self.recall_at_k(retrieved, relevant, k)
        
        if precision + recall == 0:
            return 0.0
        
        return 2 * (precision * recall) / (precision + recall)
    
    def mean_reciprocal_rank(
        self,
        retrieved: List[List[str]],
        relevant: List[List[str]]
    ) -> float:
        """Calculate Mean Reciprocal Rank (MRR)"""
        
        reciprocal_ranks = []
        
        for ret, rel in zip(retrieved, relevant):
            rel_set = set(rel)
            
            # Find rank of first relevant document
            for i, doc_id in enumerate(ret, 1):
                if doc_id in rel_set:
                    reciprocal_ranks.append(1.0 / i)
                    break
            else:
                reciprocal_ranks.append(0.0)
        
        return sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0
    
    def mean_average_precision(
        self,
        retrieved: List[List[str]],
        relevant: List[List[str]]
    ) -> float:
        """Calculate Mean Average Precision (MAP)"""
        
        average_precisions = []
        
        for ret, rel in zip(retrieved, relevant):
            rel_set = set(rel)
            
            if len(rel_set) == 0:
                average_precisions.append(0.0)
                continue
            
            # Calculate precision at each relevant position
            precisions = []
            num_relevant = 0
            
            for i, doc_id in enumerate(ret, 1):
                if doc_id in rel_set:
                    num_relevant += 1
                    precision = num_relevant / i
                    precisions.append(precision)
            
            if precisions:
                avg_precision = sum(precisions) / len(rel_set)
                average_precisions.append(avg_precision)
            else:
                average_precisions.append(0.0)
        
        return sum(average_precisions) / len(average_precisions) if average_precisions else 0.0
    
    def ndcg_at_k(
        self,
        retrieved: List[List[str]],
        relevant: List[List[str]],
        k: int
    ) -> float:
        """Calculate Normalized Discounted Cumulative Gain (NDCG@K)"""
        
        ndcg_scores = []
        
        for ret, rel in zip(retrieved, relevant):
            rel_set = set(rel)
            
            # Calculate DCG
            dcg = 0.0
            for i, doc_id in enumerate(ret[:k], 1):
                if doc_id in rel_set:
                    # Binary relevance (1 if relevant, 0 otherwise)
                    relevance = 1.0
                    dcg += relevance / (i ** 0.5)  # log2(i+1) approximation
            
            # Calculate ideal DCG
            num_relevant = min(len(rel_set), k)
            idcg = sum(1.0 / ((i + 1) ** 0.5) for i in range(num_relevant))
            
            # Calculate NDCG
            if idcg > 0:
                ndcg = dcg / idcg
                ndcg_scores.append(ndcg)
            else:
                ndcg_scores.append(0.0)
        
        return sum(ndcg_scores) / len(ndcg_scores) if ndcg_scores else 0.0
    
    def evaluate_by_source(
        self,
        queries: List[str],
        retrieved_docs: List[List[Dict[str, Any]]],
        relevant_docs: List[List[str]]
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate retrieval quality by source
        
        Args:
            queries: List of queries
            retrieved_docs: Retrieved documents with metadata per query
            relevant_docs: Ground truth relevant document IDs per query
        
        Returns:
            Metrics by source
        """
        
        # Group by source
        by_source = defaultdict(lambda: {"retrieved": [], "relevant": []})
        
        for i, (ret, rel) in enumerate(zip(retrieved_docs, relevant_docs)):
            for doc in ret:
                source = doc.get("source", "unknown")
                by_source[source]["retrieved"].append([d["id"] for d in ret if d.get("source") == source])
                by_source[source]["relevant"].append(rel)
        
        # Evaluate each source
        source_metrics = {}
        
        for source, data in by_source.items():
            if data["retrieved"]:
                metrics = self.evaluate(
                    queries=queries[:len(data["retrieved"])],
                    retrieved_docs=data["retrieved"],
                    relevant_docs=data["relevant"]
                )
                source_metrics[source] = metrics
        
        return source_metrics