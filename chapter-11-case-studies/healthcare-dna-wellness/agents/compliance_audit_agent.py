"""
Compliance Audit Agent

Ensures HIPAA compliance and maintains audit trails
"""

from typing import Dict, Any, List, Optional
import logging
from datetime import datetime
import sys
sys.path.append('../..')

from shared.base_agent import BaseAgent, AgentResponse

logger = logging.getLogger(__name__)


class ComplianceAuditAgent(BaseAgent):
    """
    Agent for HIPAA compliance and audit logging
    """
    
    def __init__(self):
        super().__init__(
            name="Compliance Audit Agent",
            model="claude-sonnet-4-20250514",
            temperature=0.0  # Deterministic for compliance
        )
        
        # Initialize audit logger
        from security.audit_trail import AuditLogger
        self.audit_logger = AuditLogger()
    
    def _get_system_prompt(self) -> str:
        """System prompt for compliance auditing"""
        return """You are a HIPAA compliance expert monitoring healthcare data access and usage.

Your role:
1. Verify all data access is authorized
2. Detect potential PHI (Protected Health Information) in outputs
3. Ensure proper audit trail logging
4. Flag compliance violations
5. Validate data handling practices

Guidelines:
- Zero tolerance for PHI leaks
- All access must be logged
- Detect and redact PHI in unexpected places
- Verify user authorization
- Flag suspicious access patterns
- Ensure minimum necessary principle

PHI includes:
- Names, dates (except year)
- Geographic subdivisions smaller than state
- Phone/fax numbers
- Email addresses
- Social Security numbers
- Medical record numbers
- Account numbers
- Biometric identifiers
- Photos
- Any other unique identifying information

Output format:
- Compliance status (PASS/FAIL)
- PHI detected (if any)
- Access authorization status
- Audit log entry confirmation
- Violations or concerns (if any)"""
    
    async def process(self, input_data: Dict[str, Any]) -> AgentResponse:
        """
        Process compliance check
        
        Args:
            input_data: {
                "action": str,
                "user_id": str,
                "patient_id": str,
                "data_accessed": Dict,
                "output_data": str (optional),
                "ip_address": str (optional),
                "authorization": Dict (optional)
            }
        
        Returns:
            AgentResponse with compliance status
        """
        action = input_data.get("action")
        user_id = input_data.get("user_id")
        patient_id = input_data.get("patient_id")
        data_accessed = input_data.get("data_accessed", {})
        output_data = input_data.get("output_data", "")
        ip_address = input_data.get("ip_address", "unknown")
        authorization = input_data.get("authorization", {})
        
        logger.info(f"Compliance check: {action} by {user_id} for patient {patient_id}")
        
        # Perform compliance checks
        compliance_results = {
            "authorization_check": await self._check_authorization(
                user_id, patient_id, action, authorization
            ),
            "phi_detection": await self._detect_phi(output_data),
            "minimum_necessary": await self._check_minimum_necessary(
                action, data_accessed
            ),
            "audit_logged": await self._log_audit_entry(
                user_id, patient_id, action, ip_address, data_accessed
            )
        }
        
        # Determine overall compliance
        compliant = all([
            compliance_results["authorization_check"]["authorized"],
            not compliance_results["phi_detection"]["phi_detected"],
            compliance_results["minimum_necessary"]["compliant"],
            compliance_results["audit_logged"]["success"]
        ])
        
        # Generate compliance report
        report = self._generate_compliance_report(compliance_results, compliant)
        
        return AgentResponse(
            content=report,
            agent_name=self.name,
            timestamp=datetime.utcnow(),
            metadata={
                "compliant": compliant,
                "compliance_results": compliance_results,
                "action": action,
                "user_id": user_id,
                "patient_id": patient_id
            },
            confidence=1.0  # Compliance is binary
        )
    
    async def _check_authorization(
        self,
        user_id: str,
        patient_id: str,
        action: str,
        authorization: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Check if user is authorized for this action"""
        
        # In production, integrate with access control system
        # For now, simplified check
        
        user_role = authorization.get("role", "unknown")
        permitted_actions = authorization.get("permitted_actions", [])
        
        authorized = action in permitted_actions
        
        return {
            "authorized": authorized,
            "user_role": user_role,
            "action": action,
            "reason": "Action permitted for role" if authorized else "Action not permitted for role"
        }
    
    async def _detect_phi(self, text: str) -> Dict[str, Any]:
        """Detect PHI in text"""
        
        import re
        
        phi_patterns = {
            "ssn": r'\b\d{3}-\d{2}-\d{4}\b',
            "phone": r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
            "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            "date": r'\b\d{1,2}/\d{1,2}/\d{4}\b',  # Dates more specific than year
            "mrn": r'\bMRN[:\s]*\d+\b',
            "account": r'\b[Aa]ccount[:\s]*\d+\b'
        }
        
        detected_phi = []
        
        for phi_type, pattern in phi_patterns.items():
            matches = re.findall(pattern, text)
            if matches:
                detected_phi.append({
                    "type": phi_type,
                    "count": len(matches),
                    "examples": matches[:3]  # First 3 examples
                })
        
        return {
            "phi_detected": len(detected_phi) > 0,
            "phi_types": detected_phi,
            "requires_redaction": len(detected_phi) > 0
        }
    
    async def _check_minimum_necessary(
        self,
        action: str,
        data_accessed: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Check if minimum necessary principle is followed
        HIPAA requires accessing only minimum data needed
        """
        
        # Define what data is necessary for each action
        necessary_data = {
            "view_genetic_data": ["patient_id", "genetic_variants"],
            "generate_recommendations": ["patient_id", "genetic_variants", "clinical_history"],
            "view_history": ["patient_id", "access_history"]
        }
        
        required = set(necessary_data.get(action, []))
        accessed = set(data_accessed.keys())
        
        # Check if accessing more than necessary
        excessive = accessed - required
        
        return {
            "compliant": len(excessive) == 0,
            "required_fields": list(required),
            "accessed_fields": list(accessed),
            "excessive_access": list(excessive)
        }
    
    async def _log_audit_entry(
        self,
        user_id: str,
        patient_id: str,
        action: str,
        ip_address: str,
        data_accessed: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Log audit entry"""
        
        try:
            audit_entry = await self.audit_logger.log_access(
                user_id=user_id,
                patient_id=patient_id,
                action=action,
                ip_address=ip_address,
                data_fields=list(data_accessed.keys()),
                timestamp=datetime.utcnow()
            )
            
            return {
                "success": True,
                "audit_entry_id": audit_entry.get("id"),
                "timestamp": audit_entry.get("timestamp")
            }
        
        except Exception as e:
            logger.error(f"Audit logging failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _generate_compliance_report(
        self,
        compliance_results: Dict[str, Any],
        compliant: bool
    ) -> str:
        """Generate human-readable compliance report"""
        
        status = "✅ COMPLIANT" if compliant else "❌ NON-COMPLIANT"
        
        report = f"""
HIPAA COMPLIANCE REPORT
{status}

Authorization: {'✅ Authorized' if compliance_results['authorization_check']['authorized'] else '❌ Not Authorized'}
- User Role: {compliance_results['authorization_check']['user_role']}
- Action: {compliance_results['authorization_check']['action']}

PHI Detection: {'✅ No PHI detected' if not compliance_results['phi_detection']['phi_detected'] else '❌ PHI detected'}
"""
        
        if compliance_results['phi_detection']['phi_detected']:
            report += "\nPHI Found:\n"
            for phi in compliance_results['phi_detection']['phi_types']:
                report += f"  - {phi['type']}: {phi['count']} instances\n"
        
        report += f"""
Minimum Necessary: {'✅ Compliant' if compliance_results['minimum_necessary']['compliant'] else '❌ Excessive access'}
"""
        
        if not compliance_results['minimum_necessary']['compliant']:
            excessive = compliance_results['minimum_necessary']['excessive_access']
            report += f"  Excessive fields accessed: {', '.join(excessive)}\n"
        
        report += f"""
Audit Logging: {'✅ Logged' if compliance_results['audit_logged']['success'] else '❌ Logging failed'}
"""
        
        if compliance_results['audit_logged']['success']:
            report += f"  Entry ID: {compliance_results['audit_logged']['audit_entry_id']}\n"
        
        return report