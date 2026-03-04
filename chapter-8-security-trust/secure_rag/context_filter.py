"""
Filter context based on user permissions.
"""

from typing import Dict, Any, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ContextFilter:
    """Filter retrieved context by user permissions"""
    
    def __init__(self, rbac_manager, pii_detector, data_anonymizer):
        """
        Initialize context filter
        
        Args:
            rbac_manager: RBAC manager
            pii_detector: PII detector
            data_anonymizer: Data anonymizer
        """
        self.rbac = rbac_manager
        self.pii_detector = pii_detector
        self.anonymizer = data_anonymizer
    
    def filter_by_permissions(self, documents: List[Dict[str, Any]],
                             user_id: str) -> List[Dict[str, Any]]:
        """
        Filter documents by user permissions
        
        Args:
            documents: Retrieved documents
            user_id: User requesting access
            
        Returns:
            Filtered documents
        """
        filtered = []
        
        for doc in documents:
            doc_id = doc.get('id', doc.get('document_id'))
            
            # Check access
            access_check = self.rbac.check_resource_access(user_id, doc_id)
            
            if access_check['allowed']:
                filtered.append(doc)
        
        return filtered
    
    def filter_pii(self, documents: List[Dict[str, Any]], user_id: str,
                  strategy: str = "mask") -> List[Dict[str, Any]]:
        """
        Filter PII from documents based on permissions
        
        Args:
            documents: Documents to filter
            user_id: User requesting access
            strategy: Anonymization strategy
            
        Returns:
            Documents with PII filtered if necessary
        """
        from ..authorization.rbac_manager import Permission
        
        can_view_pii = self.rbac.check_permission(user_id, Permission.EXPORT_DATA)
        
        if can_view_pii:
            return documents  # User can see all content
        
        filtered_docs = []
        
        for doc in documents:
            content = doc.get('content', '')
            
            # Scan for PII
            pii_scan = self.pii_detector.scan_document(content)
            
            if pii_scan['contains_pii']:
                # Anonymize
                anonymized = self.anonymizer.anonymize(content, strategy=strategy)
                
                doc_copy = doc.copy()
                doc_copy['content'] = anonymized['anonymized_text']
                doc_copy['pii_filtered'] = True
                doc_copy['pii_count'] = pii_scan['pii_count']
                
                filtered_docs.append(doc_copy)
            else:
                filtered_docs.append(doc)
        
        return filtered_docs
    
    def apply_all_filters(self, documents: List[Dict[str, Any]],
                         user_id: str) -> Dict[str, Any]:
        """
        Apply all context filters
        
        Args:
            documents: Raw retrieved documents
            user_id: User requesting access
            
        Returns:
            Filtered documents and metadata
        """
        original_count = len(documents)
        
        # 1. Filter by permissions
        permission_filtered = self.filter_by_permissions(documents, user_id)
        
        # 2. Filter PII
        pii_filtered = self.filter_pii(permission_filtered, user_id)
        
        return {
            'documents': pii_filtered,
            'original_count': original_count,
            'permission_filtered_count': original_count - len(permission_filtered),
            'final_count': len(pii_filtered),
            'filtering_applied': True
        }