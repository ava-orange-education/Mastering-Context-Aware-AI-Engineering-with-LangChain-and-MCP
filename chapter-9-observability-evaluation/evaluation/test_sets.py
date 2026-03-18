"""
Manage test datasets for evaluation.
"""

from typing import Dict, Any, List, Optional
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestSet:
    """Container for test queries and expected outputs"""
    
    def __init__(self, name: str, description: str = ""):
        """
        Initialize test set
        
        Args:
            name: Test set name
            description: Test set description
        """
        self.name = name
        self.description = description
        self.test_cases: List[Dict[str, Any]] = []
    
    def add_test_case(self, query: str, expected_answer: Optional[str] = None,
                     context: Optional[str] = None, category: Optional[str] = None,
                     metadata: Optional[Dict] = None):
        """
        Add test case to set
        
        Args:
            query: Test query
            expected_answer: Expected/correct answer
            context: Relevant context
            category: Test category
            metadata: Additional metadata
        """
        test_case = {
            'query': query,
            'expected_answer': expected_answer,
            'context': context,
            'category': category or 'general',
            'metadata': metadata or {}
        }
        
        self.test_cases.append(test_case)
        logger.info(f"Added test case to '{self.name}': {query[:50]}...")
    
    def get_queries(self) -> List[str]:
        """Get all test queries"""
        return [tc['query'] for tc in self.test_cases]
    
    def get_expected_answers(self) -> List[Optional[str]]:
        """Get all expected answers"""
        return [tc['expected_answer'] for tc in self.test_cases]
    
    def get_test_cases(self, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get test cases, optionally filtered by category
        
        Args:
            category: Filter by category
            
        Returns:
            List of test cases
        """
        if category:
            return [tc for tc in self.test_cases if tc['category'] == category]
        return self.test_cases
    
    def save(self, filepath: str):
        """Save test set to JSON file"""
        data = {
            'name': self.name,
            'description': self.description,
            'test_cases': self.test_cases
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Test set saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'TestSet':
        """Load test set from JSON file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        test_set = cls(name=data['name'], description=data.get('description', ''))
        test_set.test_cases = data['test_cases']
        
        logger.info(f"Loaded test set: {test_set.name} ({len(test_set.test_cases)} cases)")
        
        return test_set


class TestSetManager:
    """Manage multiple test sets"""
    
    def __init__(self):
        """Initialize test set manager"""
        self.test_sets: Dict[str, TestSet] = {}
    
    def create_test_set(self, name: str, description: str = "") -> TestSet:
        """Create new test set"""
        test_set = TestSet(name, description)
        self.test_sets[name] = test_set
        return test_set
    
    def add_test_set(self, test_set: TestSet):
        """Add existing test set"""
        self.test_sets[test_set.name] = test_set
    
    def get_test_set(self, name: str) -> Optional[TestSet]:
        """Get test set by name"""
        return self.test_sets.get(name)
    
    def create_rag_test_set(self) -> TestSet:
        """Create standard RAG evaluation test set"""
        test_set = self.create_test_set(
            name="rag_evaluation",
            description="Standard test set for RAG systems"
        )
        
        # Add common test cases
        test_set.add_test_case(
            query="What is the capital of France?",
            expected_answer="Paris",
            category="factual"
        )
        
        test_set.add_test_case(
            query="Explain how photosynthesis works",
            expected_answer="Photosynthesis is the process by which plants convert sunlight into energy...",
            category="explanation"
        )
        
        test_set.add_test_case(
            query="Compare machine learning and deep learning",
            expected_answer="Machine learning is a broader field, while deep learning is a subset...",
            category="comparison"
        )
        
        test_set.add_test_case(
            query="List three benefits of exercise",
            category="enumeration"
        )
        
        return test_set
    
    def create_multi_agent_test_set(self) -> TestSet:
        """Create test set for multi-agent systems"""
        test_set = self.create_test_set(
            name="multi_agent_evaluation",
            description="Test set for multi-agent coordination"
        )
        
        test_set.add_test_case(
            query="Research and summarize recent AI developments",
            category="multi_step"
        )
        
        test_set.add_test_case(
            query="Find financial data and create analysis report",
            category="coordination"
        )
        
        return test_set