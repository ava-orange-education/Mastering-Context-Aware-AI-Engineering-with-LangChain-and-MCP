"""
Entity and relation extraction for knowledge graph construction.
"""

from typing import List, Dict, Any, Tuple, Optional
import logging
import re

logger = logging.getLogger(__name__)


class Entity:
    """Represents an extracted entity"""
    
    def __init__(self, text: str, entity_type: str, start: int, end: int, metadata: Optional[Dict] = None):
        self.text = text
        self.entity_type = entity_type
        self.start = start
        self.end = end
        self.metadata = metadata or {}
        self.id = self._generate_id()
    
    def _generate_id(self) -> str:
        """Generate unique entity ID"""
        import hashlib
        content = f"{self.text}_{self.entity_type}"
        return hashlib.md5(content.encode()).hexdigest()[:16]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'id': self.id,
            'text': self.text,
            'type': self.entity_type,
            'start': self.start,
            'end': self.end,
            'metadata': self.metadata
        }


class Relation:
    """Represents a relation between entities"""
    
    def __init__(self, source: Entity, target: Entity, relation_type: str, confidence: float = 1.0):
        self.source = source
        self.target = target
        self.relation_type = relation_type
        self.confidence = confidence
        self.id = self._generate_id()
    
    def _generate_id(self) -> str:
        """Generate unique relation ID"""
        import hashlib
        content = f"{self.source.id}_{self.relation_type}_{self.target.id}"
        return hashlib.md5(content.encode()).hexdigest()[:16]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'id': self.id,
            'source': self.source.to_dict(),
            'target': self.target.to_dict(),
            'type': self.relation_type,
            'confidence': self.confidence
        }


class EntityExtractor:
    """Extract entities from text"""
    
    def __init__(self, method: str = "spacy", model_name: str = "en_core_web_sm"):
        """
        Initialize entity extractor
        
        Args:
            method: 'spacy', 'llm', or 'pattern'
            model_name: Model to use for extraction
        """
        self.method = method
        self.model_name = model_name
        
        if method == "spacy":
            self._initialize_spacy()
        elif method == "llm":
            self._initialize_llm()
    
    def _initialize_spacy(self):
        """Initialize spaCy NER"""
        try:
            import spacy
            self.nlp = spacy.load(self.model_name)
            logger.info(f"Loaded spaCy model: {self.model_name}")
        except ImportError:
            logger.error("spaCy not installed. Install with: pip install spacy")
            raise
        except OSError:
            logger.error(f"spaCy model {self.model_name} not found. Download with: python -m spacy download {self.model_name}")
            raise
    
    def _initialize_llm(self):
        """Initialize LLM for entity extraction"""
        from anthropic import Anthropic
        import os
        
        self.client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        logger.info("Initialized LLM entity extractor")
    
    def extract(self, text: str) -> List[Entity]:
        """
        Extract entities from text
        
        Args:
            text: Input text
            
        Returns:
            List of Entity objects
        """
        if self.method == "spacy":
            return self._extract_with_spacy(text)
        elif self.method == "llm":
            return self._extract_with_llm(text)
        elif self.method == "pattern":
            return self._extract_with_patterns(text)
        else:
            raise ValueError(f"Unknown extraction method: {self.method}")
    
    def _extract_with_spacy(self, text: str) -> List[Entity]:
        """Extract entities using spaCy"""
        doc = self.nlp(text)
        
        entities = []
        for ent in doc.ents:
            entity = Entity(
                text=ent.text,
                entity_type=ent.label_,
                start=ent.start_char,
                end=ent.end_char,
                metadata={'lemma': ent.lemma_}
            )
            entities.append(entity)
        
        return entities
    
    def _extract_with_llm(self, text: str) -> List[Entity]:
        """Extract entities using LLM"""
        prompt = f"""Extract all entities from the following text. For each entity, identify its type (PERSON, ORG, LOCATION, DATE, PRODUCT, etc.).

Text: {text}

Return the entities as a JSON list with format:
[{{"text": "entity text", "type": "ENTITY_TYPE", "start": start_position, "end": end_position}}]

Only return the JSON, nothing else."""
        
        response = self.client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=2048,
            messages=[{"role": "user", "content": prompt}]
        )
        
        # Parse JSON response
        import json
        try:
            response_text = response.content[0].text
            # Extract JSON from response
            json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
            if json_match:
                entities_data = json.loads(json_match.group(0))
                
                entities = []
                for ent_data in entities_data:
                    entity = Entity(
                        text=ent_data['text'],
                        entity_type=ent_data['type'],
                        start=ent_data.get('start', 0),
                        end=ent_data.get('end', 0)
                    )
                    entities.append(entity)
                
                return entities
        except Exception as e:
            logger.error(f"Error parsing LLM response: {e}")
            return []
    
    def _extract_with_patterns(self, text: str) -> List[Entity]:
        """Extract entities using regex patterns"""
        entities = []
        
        # Email pattern
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        for match in re.finditer(email_pattern, text):
            entity = Entity(
                text=match.group(0),
                entity_type='EMAIL',
                start=match.start(),
                end=match.end()
            )
            entities.append(entity)
        
        # URL pattern
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        for match in re.finditer(url_pattern, text):
            entity = Entity(
                text=match.group(0),
                entity_type='URL',
                start=match.start(),
                end=match.end()
            )
            entities.append(entity)
        
        # Phone pattern
        phone_pattern = r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'
        for match in re.finditer(phone_pattern, text):
            entity = Entity(
                text=match.group(0),
                entity_type='PHONE',
                start=match.start(),
                end=match.end()
            )
            entities.append(entity)
        
        # Date pattern (simple)
        date_pattern = r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b'
        for match in re.finditer(date_pattern, text):
            entity = Entity(
                text=match.group(0),
                entity_type='DATE',
                start=match.start(),
                end=match.end()
            )
            entities.append(entity)
        
        return entities


class RelationExtractor:
    """Extract relations between entities"""
    
    def __init__(self, method: str = "llm"):
        """
        Initialize relation extractor
        
        Args:
            method: 'llm', 'dependency', or 'pattern'
        """
        self.method = method
        
        if method == "llm":
            self._initialize_llm()
        elif method == "dependency":
            self._initialize_dependency_parser()
    
    def _initialize_llm(self):
        """Initialize LLM for relation extraction"""
        from anthropic import Anthropic
        import os
        
        self.client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        logger.info("Initialized LLM relation extractor")
    
    def _initialize_dependency_parser(self):
        """Initialize dependency parser"""
        try:
            import spacy
            self.nlp = spacy.load("en_core_web_sm")
        except ImportError:
            logger.error("spaCy not installed")
            raise
    
    def extract(self, text: str, entities: List[Entity]) -> List[Relation]:
        """
        Extract relations between entities
        
        Args:
            text: Input text
            entities: List of extracted entities
            
        Returns:
            List of Relation objects
        """
        if self.method == "llm":
            return self._extract_with_llm(text, entities)
        elif self.method == "dependency":
            return self._extract_with_dependency(text, entities)
        elif self.method == "pattern":
            return self._extract_with_patterns(text, entities)
        else:
            raise ValueError(f"Unknown extraction method: {self.method}")
    
    def _extract_with_llm(self, text: str, entities: List[Entity]) -> List[Relation]:
        """Extract relations using LLM"""
        # Format entities for prompt
        entity_list = "\n".join([f"- {e.text} ({e.entity_type})" for e in entities])
        
        prompt = f"""Given the following text and entities, extract all relationships between the entities.

Text: {text}

Entities:
{entity_list}

For each relationship, specify:
- source entity
- target entity  
- relationship type (e.g., "works_for", "located_in", "founded_by", etc.)
- confidence (0-1)

Return as JSON list:
[{{"source": "entity1", "target": "entity2", "type": "relation_type", "confidence": 0.9}}]

Only return the JSON, nothing else."""
        
        response = self.client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=2048,
            messages=[{"role": "user", "content": prompt}]
        )
        
        # Parse JSON response
        import json
        try:
            response_text = response.content[0].text
            json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
            if json_match:
                relations_data = json.loads(json_match.group(0))
                
                # Create entity lookup
                entity_lookup = {e.text: e for e in entities}
                
                relations = []
                for rel_data in relations_data:
                    source = entity_lookup.get(rel_data['source'])
                    target = entity_lookup.get(rel_data['target'])
                    
                    if source and target:
                        relation = Relation(
                            source=source,
                            target=target,
                            relation_type=rel_data['type'],
                            confidence=rel_data.get('confidence', 1.0)
                        )
                        relations.append(relation)
                
                return relations
        except Exception as e:
            logger.error(f"Error parsing LLM response: {e}")
            return []
    
    def _extract_with_dependency(self, text: str, entities: List[Entity]) -> List[Relation]:
        """Extract relations using dependency parsing"""
        doc = self.nlp(text)
        
        relations = []
        
        # Simple pattern: subject-verb-object
        for token in doc:
            if token.dep_ == "nsubj" and token.head.pos_ == "VERB":
                # Find object
                for child in token.head.children:
                    if child.dep_ in ["dobj", "pobj"]:
                        # Check if both are entities
                        source_entity = self._find_entity_at_position(entities, token.idx)
                        target_entity = self._find_entity_at_position(entities, child.idx)
                        
                        if source_entity and target_entity:
                            relation = Relation(
                                source=source_entity,
                                target=target_entity,
                                relation_type=token.head.lemma_,
                                confidence=0.8
                            )
                            relations.append(relation)
        
        return relations
    
    def _find_entity_at_position(self, entities: List[Entity], position: int) -> Optional[Entity]:
        """Find entity at given text position"""
        for entity in entities:
            if entity.start <= position < entity.end:
                return entity
        return None
    
    def _extract_with_patterns(self, text: str, entities: List[Entity]) -> List[Relation]:
        """Extract relations using patterns"""
        relations = []
        
        # Pattern: X is/was/are/were Y
        is_pattern = r'(\w+)\s+(is|was|are|were)\s+(\w+)'
        
        for match in re.finditer(is_pattern, text):
            source_text = match.group(1)
            target_text = match.group(3)
            
            # Find matching entities
            source_entity = next((e for e in entities if e.text == source_text), None)
            target_entity = next((e for e in entities if e.text == target_text), None)
            
            if source_entity and target_entity:
                relation = Relation(
                    source=source_entity,
                    target=target_entity,
                    relation_type='is_a',
                    confidence=0.7
                )
                relations.append(relation)
        
        return relations