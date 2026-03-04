"""
Secure retrieval with access control and query validation.
"""

from typing import Dict, Any, List, Optional
import logging
from ..authorization.rbac_manager import RBACManager, Permission
from ..data_protection.pii_detector import PIIDetector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SecureRetriever:
    """Retrieval with security and access control"""
    
    def __init__(self, vector_store, rbac_manager: RBACManager, 
                 pii_detector: Optional[PIIDetector] = None):
        """
        Initialize secure retriever
        
        Args:
            vector_store: Vector database
            rbac_manager: RBAC manager
            pii_detector: PII detector
        """
        self.vector_store = vector_store
        self.rbac = rbac_manager
        self.pii_detector = pii_detector or PIIDetector()
    
    def retrieve(self, query: str, user_id: str, top_k: int = 5,
                filter_metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Retrieve documents with access control
        
        Args:
            query: Search query
            user_id: User ID making request
            top_k: Number of results
            filter_metadata: Additional filters
            
        Returns:
            Retrieval results with access control applied
        """
        # Validate query
        query_validation = self._validate_query(query)
        
        if not query_validation['valid']:
            return {
                'success': False,
                'error': query_validation['reason'],
                'results': []
            }
        
        # Check if user has retrieval permission
        if not self.rbac.check_permission(user_id, Permission.EXECUTE_QUERY):
            logger.warning(f"User {user_id} lacks EXECUTE_QUERY permission")
            return {
                'success': False,
                'error': 'Insufficient permissions for query execution',
                'results': []
            }
        
        # Retrieve documents
        try:
            from ..embedding.embedding_manager import EmbeddingManager
            
            # This is a simplified example - in production, inject embedding manager
            # For now, assume vector_store.search accepts query text directly
            raw_results = self.vector_store.search(
                query_vector=query,  # In production, embed query first
                top_k=top_k * 2,  # Retrieve more, filter by access
                filter=filter_metadata
            )
            
        except Exception as e:
            logger.error(f"Retrieval error: {e}")
            return {
                'success': False,
                'error': f"Retrieval failed: {str(e)}",
                'results': []
            }
        
        # Filter results by access control
        accessible_results = []
        
        for result in raw_results:
            doc_id = result.get('id', result.get('document_id'))
            
            # Check resource access
            access_check = self.rbac.check_resource_access(user_id, doc_id)
            
            if access_check['allowed']:
                # Apply PII filtering if needed
                filtered_result = self._filter_pii(result, user_id)
                accessible_results.append(filtered_result)
            
            if len(accessible_results) >= top_k:
                break
        
        logger.info(f"Retrieved {len(accessible_results)} accessible documents for user {user_id}")
        
        return {
            'success': True,
            'results': accessible_results,
            'total_results': len(accessible_results),
            'filtered_count': len(raw_results) - len(accessible_results)
        }
    
    def _validate_query(self, query: str) -> Dict[str, bool]:
        """
        Validate query for security issues
        
        Args:
            query: Query text
            
        Returns:
            Validation result
        """
        # Check for SQL injection attempts
        sql_patterns = ['DROP', 'DELETE', 'INSERT', 'UPDATE', '--', ';']
        for pattern in sql_patterns:
            if pattern in query.upper():
                return {
                    'valid': False,
                    'reason': 'Query contains potentially malicious patterns'
                }
        
        # Check for excessively long queries
        if len(query) > 5000:
            return {
                'valid': False,
                'reason': 'Query exceeds maximum length'
            }
        
        # Check for PII in query (might indicate data exfiltration attempt)
        pii_scan = self.pii_detector.scan_document(query, return_summary=False)
        
        if pii_scan['contains_pii']:
            logger.warning(f"Query contains PII: {pii_scan['pii_count']} instances")
            # Still valid but log for monitoring
        
        return {'valid': True, 'reason': 'Query validated'}
    
    def _filter_pii(self, result: Dict[str, Any], user_id: str) -> Dict[str, Any]:
        """
        Filter PII from result based on user permissions
        
        Args:
            result: Search result
            user_id: User ID
            
        Returns:
            Filtered result
        """
        # Check if user has permission to view PII
        can_view_pii = self.rbac.check_permission(user_id, Permission.EXPORT_DATA)
        
        if can_view_pii:
            # User can see full content
            return result
        
        # Filter PII from content
        content = result.get('content', result.get('text', ''))
        
        from ..data_protection.data_anonymizer import DataAnonymizer
        anonymizer = DataAnonymizer(self.pii_detector)
        
        anonymized = anonymizer.anonymize(content, strategy="redact")
        
        # Replace content with anonymized version
        filtered_result = result.copy()
        if 'content' in filtered_result:
            filtered_result['content'] = anonymized['anonymized_text']
        if 'text' in filtered_result:
            filtered_result['text'] = anonymized['anonymized_text']
        
        filtered_result['pii_filtered'] = anonymized['pii_detected']
        
        return filtered_result