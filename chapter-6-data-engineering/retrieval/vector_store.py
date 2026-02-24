"""
Vector database integration for similarity search.
"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import logging

logger = logging.getLogger(__name__)


class VectorStore:
    """Base class for vector stores"""
    
    def __init__(self, collection_name: str, dimensions: int):
        self.collection_name = collection_name
        self.dimensions = dimensions
    
    def add_vectors(self, vectors: List[np.ndarray], metadata: List[Dict[str, Any]], ids: Optional[List[str]] = None):
        """Add vectors to the store"""
        raise NotImplementedError
    
    def search(self, query_vector: np.ndarray, top_k: int = 10, filter: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """Search for similar vectors"""
        raise NotImplementedError
    
    def delete(self, ids: List[str]):
        """Delete vectors by ID"""
        raise NotImplementedError


class ChromaVectorStore(VectorStore):
    """ChromaDB vector store implementation"""
    
    def __init__(self, collection_name: str, dimensions: int, persist_directory: str = "./chroma_db"):
        super().__init__(collection_name, dimensions)
        
        try:
            import chromadb
            
            self.client = chromadb.PersistentClient(path=persist_directory)
            
            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata={"dimensions": dimensions}
            )
            
            logger.info(f"Initialized ChromaDB collection: {collection_name}")
        
        except ImportError:
            logger.error("chromadb not installed. Install with: pip install chromadb")
            raise
    
    def add_vectors(self, vectors: List[np.ndarray], metadata: List[Dict[str, Any]], ids: Optional[List[str]] = None):
        """Add vectors to ChromaDB"""
        if ids is None:
            ids = [f"doc_{i}" for i in range(len(vectors))]
        
        # Convert numpy arrays to lists
        embeddings = [vec.tolist() if isinstance(vec, np.ndarray) else vec for vec in vectors]
        
        # Extract documents (texts) from metadata
        documents = [m.get('text', '') for m in metadata]
        
        self.collection.add(
            embeddings=embeddings,
            documents=documents,
            metadatas=metadata,
            ids=ids
        )
        
        logger.info(f"Added {len(vectors)} vectors to {self.collection_name}")
    
    def search(self, query_vector: np.ndarray, top_k: int = 10, filter: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """Search ChromaDB for similar vectors"""
        query_embedding = query_vector.tolist() if isinstance(query_vector, np.ndarray) else query_vector
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=filter
        )
        
        # Format results
        formatted_results = []
        
        for i in range(len(results['ids'][0])):
            formatted_results.append({
                'id': results['ids'][0][i],
                'document': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'distance': results['distances'][0][i] if 'distances' in results else None
            })
        
        return formatted_results
    
    def delete(self, ids: List[str]):
        """Delete vectors from ChromaDB"""
        self.collection.delete(ids=ids)
        logger.info(f"Deleted {len(ids)} vectors from {self.collection_name}")


class PineconeVectorStore(VectorStore):
    """Pinecone vector store implementation"""
    
    def __init__(self, collection_name: str, dimensions: int, api_key: str, environment: str):
        super().__init__(collection_name, dimensions)
        
        try:
            from pinecone import Pinecone, ServerlessSpec
            
            self.pc = Pinecone(api_key=api_key)
            
            # Create index if it doesn't exist
            if collection_name not in self.pc.list_indexes().names():
                self.pc.create_index(
                    name=collection_name,
                    dimension=dimensions,
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud="aws",
                        region=environment
                    )
                )
            
            self.index = self.pc.Index(collection_name)
            
            logger.info(f"Initialized Pinecone index: {collection_name}")
        
        except ImportError:
            logger.error("pinecone not installed. Install with: pip install pinecone-client")
            raise
    
    def add_vectors(self, vectors: List[np.ndarray], metadata: List[Dict[str, Any]], ids: Optional[List[str]] = None):
        """Add vectors to Pinecone"""
        if ids is None:
            ids = [f"doc_{i}" for i in range(len(vectors))]
        
        # Prepare vectors for upsert
        vectors_to_upsert = []
        
        for i, (vec, meta) in enumerate(zip(vectors, metadata)):
            vector_list = vec.tolist() if isinstance(vec, np.ndarray) else vec
            
            vectors_to_upsert.append({
                "id": ids[i],
                "values": vector_list,
                "metadata": meta
            })
        
        # Upsert in batches
        batch_size = 100
        for i in range(0, len(vectors_to_upsert), batch_size):
            batch = vectors_to_upsert[i:i + batch_size]
            self.index.upsert(vectors=batch)
        
        logger.info(f"Added {len(vectors)} vectors to {self.collection_name}")
    
    def search(self, query_vector: np.ndarray, top_k: int = 10, filter: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """Search Pinecone for similar vectors"""
        query_list = query_vector.tolist() if isinstance(query_vector, np.ndarray) else query_vector
        
        results = self.index.query(
            vector=query_list,
            top_k=top_k,
            filter=filter,
            include_metadata=True
        )
        
        # Format results
        formatted_results = []
        
        for match in results['matches']:
            formatted_results.append({
                'id': match['id'],
                'score': match['score'],
                'metadata': match.get('metadata', {})
            })
        
        return formatted_results
    
    def delete(self, ids: List[str]):
        """Delete vectors from Pinecone"""
        self.index.delete(ids=ids)
        logger.info(f"Deleted {len(ids)} vectors from {self.collection_name}")


class WeaviateVectorStore(VectorStore):
    """Weaviate vector store implementation"""
    
    def __init__(self, collection_name: str, dimensions: int, url: str = "http://localhost:8080"):
        super().__init__(collection_name, dimensions)
        
        try:
            import weaviate
            
            self.client = weaviate.Client(url)
            
            # Create class schema if it doesn't exist
            class_name = collection_name.capitalize()
            
            if not self.client.schema.exists(class_name):
                class_obj = {
                    "class": class_name,
                    "vectorizer": "none",
                    "properties": [
                        {
                            "name": "text",
                            "dataType": ["text"]
                        },
                        {
                            "name": "metadata",
                            "dataType": ["text"]
                        }
                    ]
                }
                self.client.schema.create_class(class_obj)
            
            self.class_name = class_name
            
            logger.info(f"Initialized Weaviate class: {class_name}")
        
        except ImportError:
            logger.error("weaviate-client not installed. Install with: pip install weaviate-client")
            raise
    
    def add_vectors(self, vectors: List[np.ndarray], metadata: List[Dict[str, Any]], ids: Optional[List[str]] = None):
        """Add vectors to Weaviate"""
        import json
        
        with self.client.batch as batch:
            for i, (vec, meta) in enumerate(zip(vectors, metadata)):
                vector_list = vec.tolist() if isinstance(vec, np.ndarray) else vec
                
                data_object = {
                    "text": meta.get('text', ''),
                    "metadata": json.dumps(meta)
                }
                
                batch.add_data_object(
                    data_object=data_object,
                    class_name=self.class_name,
                    vector=vector_list,
                    uuid=ids[i] if ids else None
                )
        
        logger.info(f"Added {len(vectors)} vectors to {self.class_name}")
    
    def search(self, query_vector: np.ndarray, top_k: int = 10, filter: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """Search Weaviate for similar vectors"""
        import json
        
        query_list = query_vector.tolist() if isinstance(query_vector, np.ndarray) else query_vector
        
        result = (
            self.client.query
            .get(self.class_name, ["text", "metadata"])
            .with_near_vector({"vector": query_list})
            .with_limit(top_k)
            .do()
        )
        
        # Format results
        formatted_results = []
        
        if "data" in result and "Get" in result["data"]:
            objects = result["data"]["Get"][self.class_name]
            
            for obj in objects:
                metadata = json.loads(obj.get("metadata", "{}"))
                
                formatted_results.append({
                    'text': obj.get('text', ''),
                    'metadata': metadata
                })
        
        return formatted_results
    
    def delete(self, ids: List[str]):
        """Delete vectors from Weaviate"""
        for uuid in ids:
            self.client.data_object.delete(uuid, self.class_name)
        
        logger.info(f"Deleted {len(ids)} vectors from {self.class_name}")