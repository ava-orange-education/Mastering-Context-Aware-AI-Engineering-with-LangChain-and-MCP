"""
Specialized document agent for MCP system.
"""

from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)


class DocumentAgent:
    """Agent specialized in document processing tasks"""
    
    def __init__(self, api_key: str):
        from text import DocumentProcessor
        from llm_integration import ClaudeMultimodal
        
        self.doc_processor = DocumentProcessor()
        self.claude = ClaudeMultimodal(api_key)
        
        self.capabilities = [
            'text_extraction',
            'document_qa',
            'document_summarization',
            'document_classification'
        ]
    
    def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle document-related request
        
        Args:
            request: Request dictionary with 'task' and parameters
            
        Returns:
            Result dictionary
        """
        task = request.get('task')
        
        if task == 'text_extraction':
            return self._extract_text(request)
        elif task == 'document_qa':
            return self._answer_document_questions(request)
        elif task == 'document_summarization':
            return self._summarize_document(request)
        elif task == 'document_classification':
            return self._classify_document(request)
        else:
            return {'error': f'Unknown task: {task}'}
    
    def _extract_text(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Extract text from document"""
        doc_path = request['document_path']
        
        result = self.doc_processor.process_document(doc_path)
        
        return {
            'task': 'text_extraction',
            'document_path': doc_path,
            'extracted_text': result.get('text'),
            'error': result.get('error')
        }
    
    def _answer_document_questions(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Answer questions about document"""
        doc_path = request['document_path']
        questions = request['questions']
        
        answers = self.claude.document_analysis(doc_path, questions)
        
        return {
            'task': 'document_qa',
            'document_path': doc_path,
            'questions': questions,
            'answers': answers
        }
    
    def _summarize_document(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Summarize document"""
        doc_path = request['document_path']
        
        # Extract text first
        doc_info = self.doc_processor.process_document(doc_path)
        text = doc_info.get('text', '')
        
        if not text:
            return {'error': 'Could not extract text from document'}
        
        # Use Claude to summarize (treating as image for PDFs)
        summary = self.claude.analyze_image(
            doc_path,
            "Provide a comprehensive summary of this document."
        )
        
        return {
            'task': 'document_summarization',
            'document_path': doc_path,
            'summary': summary
        }
    
    def _classify_document(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Classify document type"""
        doc_path = request['document_path']
        categories = request.get('categories', ['invoice', 'contract', 'report', 'letter'])
        
        prompt = f"Classify this document into one of these categories: {', '.join(categories)}. Respond with just the category name."
        
        classification = self.claude.analyze_image(doc_path, prompt)
        
        return {
            'task': 'document_classification',
            'document_path': doc_path,
            'classification': classification.strip(),
            'categories': categories
        }