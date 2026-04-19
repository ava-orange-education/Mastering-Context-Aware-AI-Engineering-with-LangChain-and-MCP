"""
Tests for HIPAA compliance features
"""

import pytest
import sys
sys.path.append('../..')

from security.access_control import AccessControl, Role, Permission
from security.audit_trail import AuditLogger
from security.patient_data_encryption import encrypt_patient_data, decrypt_patient_data
from integrations.hipaa_logger import HIPAALogger


class TestAccessControl:
    """Test access control system"""
    
    def test_role_permissions(self):
        """Test role-based permissions"""
        ac = AccessControl()
        
        # Physicians should have view and edit permissions
        physician_perms = ac.get_user_permissions(Role.PHYSICIAN)
        assert Permission.VIEW_GENETIC_DATA in physician_perms
        assert Permission.EDIT_CLINICAL_DATA in physician_perms
        assert Permission.GENERATE_RECOMMENDATIONS in physician_perms
        
        # Patients should only have view permissions
        patient_perms = ac.get_user_permissions(Role.PATIENT)
        assert Permission.VIEW_GENETIC_DATA in patient_perms
        assert Permission.EDIT_GENETIC_DATA not in patient_perms
    
    def test_patient_own_data_access(self):
        """Test patients can only access their own data"""
        ac = AccessControl()
        
        # Patient accessing own data - should succeed
        allowed = ac.check_permission(
            user_id="P123",
            user_role=Role.PATIENT,
            permission=Permission.VIEW_GENETIC_DATA,
            patient_id="P123"
        )
        assert allowed is True
        
        # Patient accessing other's data - should fail
        denied = ac.check_permission(
            user_id="P123",
            user_role=Role.PATIENT,
            permission=Permission.VIEW_GENETIC_DATA,
            patient_id="P456"
        )
        assert denied is False
    
    def test_emergency_access(self):
        """Test emergency break-glass access"""
        ac = AccessControl()
        
        # Grant emergency access
        ac.grant_emergency_access(
            user_id="doctor123",
            patient_id="P789",
            duration_minutes=60
        )
        
        # Should have access
        has_access = ac._has_emergency_access("doctor123", "P789")
        assert has_access is True
    
    def test_validate_data_access(self):
        """Test data access validation"""
        ac = AccessControl()
        
        access_results = ac.validate_data_access(
            user_id="doctor123",
            user_role=Role.PHYSICIAN,
            patient_id="P123",
            data_types=["genetic_data", "clinical_data", "demographics"]
        )
        
        assert access_results["genetic_data"] is True
        assert access_results["clinical_data"] is True
        assert access_results["demographics"] is True


@pytest.mark.asyncio
class TestAuditTrail:
    """Test audit trail logging"""
    
    async def test_log_access(self):
        """Test logging patient data access"""
        audit = AuditLogger()
        
        entry = await audit.log_access(
            user_id="doctor123",
            patient_id="P123",
            action="view_genetic_data",
            ip_address="192.168.1.100",
            data_fields=["genetic_variants", "lab_results"]
        )
        
        assert entry["id"] is not None
        assert entry["event_type"] == "patient_access"
        assert entry["user_id"] == "doctor123"
        assert entry["patient_id"] == "P123"
        assert entry["success"] is True
    
    async def test_log_modification(self):
        """Test logging data modifications"""
        audit = AuditLogger()
        
        entry = await audit.log_modification(
            user_id="doctor123",
            patient_id="P123",
            resource="patient_record",
            field="email",
            old_value="old@example.com",
            new_value="new@example.com",
            reason="Patient requested update"
        )
        
        assert entry["event_type"] == "data_modification"
        assert entry["field"] == "email"
    
    async def test_get_patient_access_log(self):
        """Test retrieving patient access log"""
        audit = AuditLogger()
        
        # Log some accesses
        await audit.log_access(
            user_id="doctor123",
            patient_id="P123",
            action="view_genetic_data",
            ip_address="192.168.1.100"
        )
        
        await audit.log_access(
            user_id="nurse456",
            patient_id="P123",
            action="view_demographics",
            ip_address="192.168.1.101"
        )
        
        # Retrieve log
        log = await audit.get_patient_access_log("P123", days=30)
        
        assert len(log) >= 2
        assert all(entry["patient_id"] == "P123" for entry in log)
    
    async def test_search_audit_log(self):
        """Test searching audit log"""
        audit = AuditLogger()
        
        # Log access
        await audit.log_access(
            user_id="doctor123",
            patient_id="P123",
            action="view_genetic_data",
            ip_address="192.168.1.100"
        )
        
        # Search by user
        results = await audit.search_audit_log(
            filters={"user_id": "doctor123"},
            limit=10
        )
        
        assert len(results) > 0
        assert all(r["user_id"] == "doctor123" for r in results)


class TestEncryption:
    """Test patient data encryption"""
    
    def test_encrypt_decrypt_data(self):
        """Test encrypting and decrypting patient data"""
        
        patient_data = {
            "patient_id": "P123",
            "name": "John Doe",
            "genetic_variants": [
                {"gene": "MTHFR", "variant": "C677T"}
            ]
        }
        
        # Encrypt
        encrypted = encrypt_patient_data(patient_data)
        
        assert isinstance(encrypted, str)
        assert encrypted != str(patient_data)
        
        # Decrypt
        decrypted = decrypt_patient_data(encrypted)
        
        assert decrypted == patient_data
    
    def test_encrypt_different_data_different_output(self):
        """Test that different data produces different ciphertext"""
        
        data1 = {"patient_id": "P123"}
        data2 = {"patient_id": "P456"}
        
        encrypted1 = encrypt_patient_data(data1)
        encrypted2 = encrypt_patient_data(data2)
        
        assert encrypted1 != encrypted2


class TestHIPAALogger:
    """Test HIPAA-specific logging"""
    
    def test_log_access(self):
        """Test HIPAA access logging"""
        logger = HIPAALogger()
        
        # Should not raise exception
        logger.log_access(
            user_id="doctor123",
            patient_id="P123",
            action="view",
            resource="genetic_data",
            ip_address="192.168.1.100",
            success=True
        )
    
    def test_log_phi_access(self):
        """Test PHI access logging"""
        logger = HIPAALogger()
        
        logger.log_phi_access(
            user_id="doctor123",
            phi_type="genetic",
            purpose="treatment",
            fields_accessed=["genetic_variants", "lab_results"],
            justification="Patient consultation"
        )
    
    def test_log_consent(self):
        """Test consent logging"""
        logger = HIPAALogger()
        
        logger.log_consent(
            patient_id="P123",
            consent_type="genetic_testing",
            granted=True,
            scope="wellness_recommendations",
            expiration="2025-12-31"
        )
    
    def test_log_breach(self):
        """Test security breach logging"""
        logger = HIPAALogger()
        
        logger.log_breach(
            breach_type="unauthorized_access",
            affected_patients=10,
            description="Unauthorized access attempt detected",
            severity="high",
            mitigation="Access revoked, passwords reset"
        )
    
    def test_phi_redaction(self):
        """Test PHI redaction"""
        logger = HIPAALogger()
        
        data_with_phi = "Patient SSN: 123-45-6789, Phone: 555-1234"
        redacted = logger._redact_phi(data_with_phi)
        
        assert "123-45-6789" not in redacted
        assert "555-1234" not in redacted
        assert "REDACTED" in redacted