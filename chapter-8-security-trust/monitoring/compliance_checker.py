"""
Compliance validation and reporting.
"""

from typing import Dict, Any, List
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ComplianceChecker:
    """Check compliance with security and privacy regulations"""
    
    def __init__(self):
        """Initialize compliance checker"""
        self.compliance_rules = {
            'gdpr': {
                'name': 'GDPR',
                'requires_audit_logs': True,
                'requires_pii_protection': True,
                'data_retention_days': 365,
                'requires_consent': True
            },
            'hipaa': {
                'name': 'HIPAA',
                'requires_audit_logs': True,
                'requires_encryption': True,
                'requires_access_control': True,
                'data_retention_days': 2555  # 7 years
            },
            'pci_dss': {
                'name': 'PCI DSS',
                'requires_encryption': True,
                'requires_access_control': True,
                'requires_logging': True,
                'log_retention_days': 90
            }
        }
    
    def check_audit_log_compliance(self, audit_logs: List[Any], 
                                   regulation: str = 'gdpr') -> Dict[str, Any]:
        """
        Check audit log compliance
        
        Args:
            audit_logs: List of audit logs
            regulation: Regulation to check against
            
        Returns:
            Compliance status
        """
        rule = self.compliance_rules.get(regulation, {})
        
        if not rule.get('requires_audit_logs'):
            return {'compliant': True, 'reason': 'Audit logs not required'}
        
        # Check if sufficient logs exist
        if not audit_logs:
            return {
                'compliant': False,
                'reason': f'{rule["name"]} requires audit logs but none found'
            }
        
        # Check retention period
        retention_days = rule.get('data_retention_days', 365)
        oldest_required = datetime.now() - timedelta(days=retention_days)
        
        oldest_log = min(audit_logs, key=lambda x: x.timestamp)
        
        if oldest_log.timestamp > oldest_required:
            return {
                'compliant': False,
                'reason': f'Audit logs must be retained for {retention_days} days',
                'oldest_log_age': (datetime.now() - oldest_log.timestamp).days
            }
        
        return {
            'compliant': True,
            'reason': f'Audit logs meet {rule["name"]} requirements',
            'log_count': len(audit_logs)
        }
    
    def check_pii_protection(self, pii_accesses: List[Any]) -> Dict[str, Any]:
        """
        Check PII protection compliance
        
        Args:
            pii_accesses: PII access events
            
        Returns:
            Compliance status
        """
        if not pii_accesses:
            return {'compliant': True, 'reason': 'No PII accessed'}
        
        # Check if all PII accesses are logged
        unlogged_access = [a for a in pii_accesses if not hasattr(a, 'timestamp')]
        
        if unlogged_access:
            return {
                'compliant': False,
                'reason': 'Some PII accesses not properly logged',
                'unlogged_count': len(unlogged_access)
            }
        
        # Check if access control was enforced
        unauthorized_access = [
            a for a in pii_accesses 
            if a.metadata.get('permission_checked') == False
        ]
        
        if unauthorized_access:
            return {
                'compliant': False,
                'reason': 'PII accessed without permission checks',
                'violations': len(unauthorized_access)
            }
        
        return {
            'compliant': True,
            'reason': 'PII protection measures in place',
            'total_accesses': len(pii_accesses)
        }
    
    def check_encryption_compliance(self, encryption_enabled: bool) -> Dict[str, Any]:
        """Check encryption compliance"""
        if not encryption_enabled:
            return {
                'compliant': False,
                'reason': 'Data encryption is required but not enabled'
            }
        
        return {
            'compliant': True,
            'reason': 'Encryption is enabled'
        }
    
    def generate_compliance_report(self, regulation: str,
                                   audit_logs: List[Any],
                                   pii_accesses: List[Any],
                                   encryption_enabled: bool) -> Dict[str, Any]:
        """
        Generate comprehensive compliance report
        
        Args:
            regulation: Regulation to check
            audit_logs: Audit log entries
            pii_accesses: PII access events
            encryption_enabled: Whether encryption is enabled
            
        Returns:
            Compliance report
        """
        rule = self.compliance_rules.get(regulation)
        
        if not rule:
            return {'error': f'Unknown regulation: {regulation}'}
        
        checks = []
        all_compliant = True
        
        # Check audit logs
        if rule.get('requires_audit_logs'):
            audit_check = self.check_audit_log_compliance(audit_logs, regulation)
            checks.append({
                'check': 'Audit Logs',
                'compliant': audit_check['compliant'],
                'details': audit_check
            })
            all_compliant = all_compliant and audit_check['compliant']
        
        # Check PII protection
        if rule.get('requires_pii_protection'):
            pii_check = self.check_pii_protection(pii_accesses)
            checks.append({
                'check': 'PII Protection',
                'compliant': pii_check['compliant'],
                'details': pii_check
            })
            all_compliant = all_compliant and pii_check['compliant']
        
        # Check encryption
        if rule.get('requires_encryption'):
            enc_check = self.check_encryption_compliance(encryption_enabled)
            checks.append({
                'check': 'Data Encryption',
                'compliant': enc_check['compliant'],
                'details': enc_check
            })
            all_compliant = all_compliant and enc_check['compliant']
        
        return {
            'regulation': rule['name'],
            'overall_compliant': all_compliant,
            'checks': checks,
            'generated_at': datetime.now().isoformat()
        }