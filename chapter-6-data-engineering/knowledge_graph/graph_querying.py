"""
Graph querying and traversal utilities.
"""

from typing import List, Dict, Any, Optional, Set
import logging
from .graph_builder import KnowledgeGraph
from .entity_extractor import Entity, Relation

logger = logging.getLogger(__name__)


class GraphQuery:
    """Query interface for knowledge graph"""
    
    def __init__(self, graph: KnowledgeGraph):
        self.graph = graph
    
    def find_path(self, source_id: str, target_id: str, max_depth: int = 3) -> Optional[List[Entity]]:
        """
        Find shortest path between two entities
        
        Args:
            source_id: Source entity ID
            target_id: Target entity ID
            max_depth: Maximum path length
            
        Returns:
            List of entities in path, or None if no path found
        """
        from collections import deque
        
        # BFS to find shortest path
        queue = deque([(source_id, [source_id])])
        visited = {source_id}
        
        while queue:
            current_id, path = queue.popleft()
            
            if len(path) > max_depth:
                continue
            
            if current_id == target_id:
                # Found path, convert IDs to entities
                return [self.graph.get_entity(eid) for eid in path]
            
            # Get neighbors
            neighbors = self.graph.get_neighbors(current_id)
            
            for neighbor in neighbors:
                if neighbor.id not in visited:
                    visited.add(neighbor.id)
                    queue.append((neighbor.id, path + [neighbor.id]))
        
        return None
    
    def find_related_entities(self, entity_id: str, relation_types: Optional[List[str]] = None, max_depth: int = 2) -> List[Entity]:
        """
        Find entities related to a given entity
        
        Args:
            entity_id: Starting entity ID
            relation_types: Optional filter by relation types
            max_depth: Maximum traversal depth
            
        Returns:
            List of related entities
        """
        from collections import deque
        
        related = set()
        queue = deque([(entity_id, 0)])
        visited = {entity_id}
        
        while queue:
            current_id, depth = queue.popleft()
            
            if depth >= max_depth:
                continue
            
            # Get all relations for current entity
            relations = self.graph.get_relations_for_entity(current_id)
            
            for relation in relations:
                # Check relation type filter
                if relation_types and relation.relation_type not in relation_types:
                    continue
                
                # Get the other entity in the relation
                if relation.source.id == current_id:
                    other_entity = relation.target
                else:
                    other_entity = relation.source
                
                if other_entity.id not in visited:
                    visited.add(other_entity.id)
                    related.add(other_entity.id)
                    queue.append((other_entity.id, depth + 1))
        
        return [self.graph.get_entity(eid) for eid in related]
    
    def find_clusters(self, min_cluster_size: int = 3) -> List[List[Entity]]:
        """
        Find clusters of highly connected entities
        
        Args:
            min_cluster_size: Minimum entities per cluster
            
        Returns:
            List of clusters (each cluster is a list of entities)
        """
        visited = set()
        clusters = []
        
        for entity_id in self.graph.entities:
            if entity_id in visited:
                continue
            
            # BFS to find connected component
            cluster = self._get_connected_component(entity_id, visited)
            
            if len(cluster) >= min_cluster_size:
                clusters.append([self.graph.get_entity(eid) for eid in cluster])
        
        return clusters
    
    def _get_connected_component(self, start_id: str, visited: Set[str]) -> Set[str]:
        """Get all entities in connected component"""
        from collections import deque
        
        component = set()
        queue = deque([start_id])
        visited.add(start_id)
        
        while queue:
            current_id = queue.popleft()
            component.add(current_id)
            
            neighbors = self.graph.get_neighbors(current_id)
            
            for neighbor in neighbors:
                if neighbor.id not in visited:
                    visited.add(neighbor.id)
                    queue.append(neighbor.id)
        
        return component
    
    def get_subgraph(self, entity_ids: List[str], include_relations: bool = True) -> Dict[str, Any]:
        """
        Extract subgraph containing specific entities
        
        Args:
            entity_ids: List of entity IDs to include
            include_relations: Whether to include relations between entities
            
        Returns:
            Subgraph dictionary
        """
        entities = [self.graph.get_entity(eid) for eid in entity_ids if self.graph.get_entity(eid)]
        
        relations = []
        if include_relations:
            entity_id_set = set(entity_ids)
            
            for relation in self.graph.relations:
                if relation.source.id in entity_id_set and relation.target.id in entity_id_set:
                    relations.append(relation)
        
        return {
            'entities': [e.to_dict() for e in entities],
            'relations': [r.to_dict() for r in relations]
        }
    
    def query_by_pattern(self, pattern: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Query graph by pattern matching
        
        Args:
            pattern: Pattern dictionary with 'source_type', 'relation_type', 'target_type'
            
        Returns:
            List of matches
        """
        matches = []
        
        source_type = pattern.get('source_type')
        relation_type = pattern.get('relation_type')
        target_type = pattern.get('target_type')
        
        for relation in self.graph.relations:
            # Check if relation matches pattern
            if relation_type and relation.relation_type != relation_type:
                continue
            
            if source_type and relation.source.entity_type != source_type:
                continue
            
            if target_type and relation.target.entity_type != target_type:
                continue
            
            matches.append({
                'source': relation.source.to_dict(),
                'relation': relation.relation_type,
                'target': relation.target.to_dict(),
                'confidence': relation.confidence
            })
        
        return matches