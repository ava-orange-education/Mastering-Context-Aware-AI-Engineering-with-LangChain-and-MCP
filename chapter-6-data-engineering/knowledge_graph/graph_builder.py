"""
Knowledge graph construction and management.
"""

from typing import List, Dict, Any, Optional, Set
import logging
from .entity_extractor import Entity, Relation, EntityExtractor, RelationExtractor

logger = logging.getLogger(__name__)


class KnowledgeGraph:
    """In-memory knowledge graph"""
    
    def __init__(self):
        self.entities: Dict[str, Entity] = {}
        self.relations: List[Relation] = []
        self.entity_index: Dict[str, Set[str]] = {}  # entity_type -> set of entity IDs
    
    def add_entity(self, entity: Entity):
        """Add entity to graph"""
        self.entities[entity.id] = entity
        
        # Update index
        if entity.entity_type not in self.entity_index:
            self.entity_index[entity.entity_type] = set()
        self.entity_index[entity.entity_type].add(entity.id)
    
    def add_relation(self, relation: Relation):
        """Add relation to graph"""
        # Ensure entities exist
        if relation.source.id not in self.entities:
            self.add_entity(relation.source)
        if relation.target.id not in self.entities:
            self.add_entity(relation.target)
        
        self.relations.append(relation)
    
    def get_entity(self, entity_id: str) -> Optional[Entity]:
        """Get entity by ID"""
        return self.entities.get(entity_id)
    
    def get_entities_by_type(self, entity_type: str) -> List[Entity]:
        """Get all entities of a specific type"""
        entity_ids = self.entity_index.get(entity_type, set())
        return [self.entities[eid] for eid in entity_ids]
    
    def get_relations_for_entity(self, entity_id: str) -> List[Relation]:
        """Get all relations involving an entity"""
        return [
            r for r in self.relations
            if r.source.id == entity_id or r.target.id == entity_id
        ]
    
    def get_neighbors(self, entity_id: str, relation_type: Optional[str] = None) -> List[Entity]:
        """
        Get neighboring entities
        
        Args:
            entity_id: Entity ID
            relation_type: Optional filter by relation type
            
        Returns:
            List of neighboring entities
        """
        neighbors = []
        
        for relation in self.relations:
            if relation.source.id == entity_id:
                if relation_type is None or relation.relation_type == relation_type:
                    neighbors.append(relation.target)
            elif relation.target.id == entity_id:
                if relation_type is None or relation.relation_type == relation_type:
                    neighbors.append(relation.source)
        
        return neighbors
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert graph to dictionary"""
        return {
            'entities': [e.to_dict() for e in self.entities.values()],
            'relations': [r.to_dict() for r in self.relations],
            'stats': {
                'num_entities': len(self.entities),
                'num_relations': len(self.relations),
                'entity_types': list(self.entity_index.keys())
            }
        }
    
    def export_to_neo4j(self, uri: str, username: str, password: str):
        """Export graph to Neo4j database"""
        try:
            from neo4j import GraphDatabase
            
            driver = GraphDatabase.driver(uri, auth=(username, password))
            
            with driver.session() as session:
                # Create entities
                for entity in self.entities.values():
                    session.run(
                        f"MERGE (e:{entity.entity_type} {{id: $id, text: $text}})",
                        id=entity.id,
                        text=entity.text
                    )
                
                # Create relations
                for relation in self.relations:
                    session.run(
                        f"""
                        MATCH (s:{relation.source.entity_type} {{id: $source_id}})
                        MATCH (t:{relation.target.entity_type} {{id: $target_id}})
                        MERGE (s)-[r:{relation.relation_type}]->(t)
                        SET r.confidence = $confidence
                        """,
                        source_id=relation.source.id,
                        target_id=relation.target.id,
                        confidence=relation.confidence
                    )
            
            driver.close()
            logger.info("Exported graph to Neo4j")
        
        except ImportError:
            logger.error("neo4j package not installed. Install with: pip install neo4j")
            raise


class GraphBuilder:
    """Build knowledge graph from documents"""
    
    def __init__(self, entity_method: str = "spacy", relation_method: str = "llm"):
        """
        Initialize graph builder
        
        Args:
            entity_method: Method for entity extraction
            relation_method: Method for relation extraction
        """
        self.entity_extractor = EntityExtractor(method=entity_method)
        self.relation_extractor = RelationExtractor(method=relation_method)
        self.graph = KnowledgeGraph()
    
    def build_from_text(self, text: str) -> KnowledgeGraph:
        """
        Build knowledge graph from text
        
        Args:
            text: Input text
            
        Returns:
            KnowledgeGraph object
        """
        # Extract entities
        entities = self.entity_extractor.extract(text)
        logger.info(f"Extracted {len(entities)} entities")
        
        # Add entities to graph
        for entity in entities:
            self.graph.add_entity(entity)
        
        # Extract relations
        relations = self.relation_extractor.extract(text, entities)
        logger.info(f"Extracted {len(relations)} relations")
        
        # Add relations to graph
        for relation in relations:
            self.graph.add_relation(relation)
        
        return self.graph
    
    def build_from_documents(self, documents: List[Dict[str, Any]]) -> KnowledgeGraph:
        """
        Build knowledge graph from multiple documents
        
        Args:
            documents: List of document dictionaries
            
        Returns:
            KnowledgeGraph object
        """
        for doc in documents:
            text = doc.get('content', '')
            
            if text:
                # Extract entities
                entities = self.entity_extractor.extract(text)
                
                # Add document metadata to entities
                for entity in entities:
                    entity.metadata['document_id'] = doc.get('id')
                    self.graph.add_entity(entity)
                
                # Extract relations
                relations = self.relation_extractor.extract(text, entities)
                
                for relation in relations:
                    self.graph.add_relation(relation)
        
        logger.info(f"Built graph from {len(documents)} documents")
        logger.info(f"Total entities: {len(self.graph.entities)}")
        logger.info(f"Total relations: {len(self.graph.relations)}")
        
        return self.graph
    
    def get_graph(self) -> KnowledgeGraph:
        """Get the constructed graph"""
        return self.graph