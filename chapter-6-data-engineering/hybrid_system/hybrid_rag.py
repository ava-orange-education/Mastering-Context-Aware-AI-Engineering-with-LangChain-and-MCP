"""
Complete hybrid RAG system combining vector search and knowledge graphs.
"""

from typing import List, Dict, Any, Optional
import numpy as np
import logging

logger = logging.getLogger(__name__)


class HybridRAGSystem:
    """
    Hybrid RAG system combining:
    - Dense vector retrieval
    - Sparse keyword retrieval (BM25)
    - Knowledge graph traversal
    """
    
    def __init__(self, vector_store, bm25_index, knowledge_graph, 
                 embedding_manager, llm_client):
        """
        Initialize hybrid RAG system
        
        Args:
            vector_store: Vector database
            bm25_index: BM25 index for keyword search
            knowledge_graph: Knowledge graph
            embedding_manager: Embedding generator
            llm_client: LLM client for generation
        """
        self.vector_store = vector_store
        self.bm25_index = bm25_index
        self.knowledge_graph = knowledge_graph
        self.embedding_manager = embedding_manager
        self.llm_client = llm_client
        
        # Fusion weights
        self.vector_weight = 0.5
        self.bm25_weight = 0.3
        self.graph_weight = 0.2
    
    def set_fusion_weights(self, vector: float, bm25: float, graph: float):
        """Set weights for result fusion"""
        total = vector + bm25 + graph
        self.vector_weight = vector / total
        self.bm25_weight = bm25 / total
        self.graph_weight = graph / total
        
        logger.info(f"Set fusion weights: vector={self.vector_weight:.2f}, "
                   f"bm25={self.bm25_weight:.2f}, graph={self.graph_weight:.2f}")
    
    def retrieve(self, query: str, top_k: int = 10, 
                use_graph: bool = True) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents using hybrid approach
        
        Args:
            query: Query string
            top_k: Number of results to return
            use_graph: Whether to include graph-based retrieval
            
        Returns:
            List of retrieved documents with scores
        """
        # Generate query embedding
        query_embedding = self.embedding_manager.embed_text(query).embedding
        
        # Dense retrieval (vector search)
        vector_results = self.vector_store.search(
            query_vector=query_embedding,
            top_k=top_k * 2
        )
        
        # Sparse retrieval (BM25)
        bm25_results = self.bm25_index.search(query, top_k=top_k * 2)
        
        # Graph-based retrieval
        if use_graph:
            graph_results = self._graph_retrieval(query, top_k=top_k)
        else:
            graph_results = []
        
        # Fuse results
        fused_results = self._fuse_results(
            vector_results, bm25_results, graph_results, top_k
        )
        
        return fused_results
    
    def _graph_retrieval(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Retrieve using knowledge graph"""
        from .graph_querying import GraphQuery
        
        # Extract entities from query
        from ..knowledge_graph.entity_extractor import EntityExtractor
        entity_extractor = EntityExtractor(method="spacy")
        query_entities = entity_extractor.extract(query)
        
        if not query_entities:
            return []
        
        graph_query = GraphQuery(self.knowledge_graph)
        
        # Find related entities for each query entity
        related_documents = []
        
        for entity in query_entities:
            # Find entity in graph
            matching_entities = [
                e for e in self.knowledge_graph.entities.values()
                if e.text.lower() == entity.text.lower()
            ]
            
            if not matching_entities:
                continue
            
            graph_entity = matching_entities[0]
            
            # Get related entities
            related = graph_query.find_related_entities(
                entity_id=graph_entity.id,
                max_depth=2
            )
            
            # Extract associated documents
            for rel_entity in related:
                if 'document_id' in rel_entity.metadata:
                    related_documents.append({
                        'id': rel_entity.metadata['document_id'],
                        'entity': rel_entity.text,
                        'entity_type': rel_entity.entity_type,
                        'score': 1.0  # Could be refined based on graph distance
                    })
        
        # Deduplicate and score
        doc_scores = {}
        for doc in related_documents:
            doc_id = doc['id']
            if doc_id in doc_scores:
                doc_scores[doc_id]['score'] += doc['score']
            else:
                doc_scores[doc_id] = doc
        
        # Sort by score
        results = sorted(doc_scores.values(), key=lambda x: x['score'], reverse=True)
        
        return results[:top_k]
    
    def _fuse_results(self, vector_results: List[Dict], 
                     bm25_results: List[Dict],
                     graph_results: List[Dict],
                     top_k: int) -> List[Dict[str, Any]]:
        """Fuse results from multiple sources"""
        # Normalize scores
        vector_scores = self._normalize_scores(
            [r.get('distance', r.get('score', 0)) for r in vector_results]
        )
        bm25_scores = self._normalize_scores(
            [r.get('score', 0) for r in bm25_results]
        )
        graph_scores = self._normalize_scores(
            [r.get('score', 0) for r in graph_results]
        )
        
        # Create unified score dictionary
        all_scores = {}
        
        # Add vector scores
        for i, result in enumerate(vector_results):
            doc_id = result.get('id', str(i))
            all_scores[doc_id] = {
                'vector_score': vector_scores[i] * self.vector_weight,
                'bm25_score': 0,
                'graph_score': 0,
                'document': result.get('document', ''),
                'metadata': result.get('metadata', {})
            }
        
        # Add BM25 scores
        for i, result in enumerate(bm25_results):
            doc_id = result.get('id', str(i))
            
            if doc_id in all_scores:
                all_scores[doc_id]['bm25_score'] = bm25_scores[i] * self.bm25_weight
            else:
                all_scores[doc_id] = {
                    'vector_score': 0,
                    'bm25_score': bm25_scores[i] * self.bm25_weight,
                    'graph_score': 0,
                    'document': result.get('document', ''),
                    'metadata': result.get('metadata', {})
                }
        
        # Add graph scores
        for i, result in enumerate(graph_results):
            doc_id = result.get('id', str(i))
            
            if doc_id in all_scores:
                all_scores[doc_id]['graph_score'] = graph_scores[i] * self.graph_weight
            else:
                all_scores[doc_id] = {
                    'vector_score': 0,
                    'bm25_score': 0,
                    'graph_score': graph_scores[i] * self.graph_weight,
                    'document': '',
                    'metadata': {}
                }
        
        # Calculate combined scores
        for doc_id in all_scores:
            all_scores[doc_id]['combined_score'] = (
                all_scores[doc_id]['vector_score'] +
                all_scores[doc_id]['bm25_score'] +
                all_scores[doc_id]['graph_score']
            )
        
        # Sort by combined score
        sorted_results = sorted(
            all_scores.items(),
            key=lambda x: x[1]['combined_score'],
            reverse=True
        )
        
        # Format results
        final_results = []
        for doc_id, scores in sorted_results[:top_k]:
            final_results.append({
                'id': doc_id,
                'document': scores['document'],
                'metadata': scores['metadata'],
                'combined_score': scores['combined_score'],
                'vector_score': scores['vector_score'],
                'bm25_score': scores['bm25_score'],
                'graph_score': scores['graph_score']
            })
        
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
    
    def query(self, question: str, top_k: int = 5, 
             use_graph: bool = True) -> Dict[str, Any]:
        """
        Complete RAG query with retrieval and generation
        
        Args:
            question: User question
            top_k: Number of documents to retrieve
            use_graph: Whether to use graph-based retrieval
            
        Returns:
            Generated answer with sources
        """
        # Retrieve relevant documents
        retrieved_docs = self.retrieve(question, top_k=top_k, use_graph=use_graph)
        
        if not retrieved_docs:
            return {
                'answer': "I couldn't find any relevant information to answer your question.",
                'sources': [],
                'confidence': 0.0
            }
        
        # Build context from retrieved documents
        context_parts = []
        sources = []
        
        for i, doc in enumerate(retrieved_docs):
            context_parts.append(f"[{i+1}] {doc['document']}")
            sources.append({
                'index': i + 1,
                'id': doc['id'],
                'score': doc['combined_score'],
                'metadata': doc.get('metadata', {})
            })
        
        context = "\n\n".join(context_parts)
        
        # Generate answer
        prompt = f"""Answer the following question based on the provided context. If the context doesn't contain enough information to answer the question, say so.

Context:
{context}

Question: {question}

Answer:"""
        
        response = self.llm_client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}]
        )
        
        answer = response.content[0].text
        
        return {
            'answer': answer,
            'sources': sources,
            'num_sources': len(sources),
            'retrieval_method': 'hybrid' if use_graph else 'vector+bm25'
        }