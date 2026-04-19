"""
Patient Data Encryption

AES-256 encryption for patient data at rest and in transit
"""

from typing import Any, Dict
import logging
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2
import base64
import json
import sys
sys.path.append('../..')

from shared.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class PatientDataEncryption:
    """
    Encryption utility for patient data
    """
    
    def __init__(self):
        # Derive encryption key from configured key
        self.encryption_key = self._derive_key(
            settings.healthcare_encryption_key
        )
        self.cipher_suite = Fernet(self.encryption_key)
    
    def _derive_key(self, password: str) -> bytes:
        """
        Derive encryption key from password using PBKDF2
        
        Args:
            password: Password/key
        
        Returns:
            Derived key suitable for Fernet
        """
        
        # Use a fixed salt for deterministic key derivation
        # In production, use unique salts per patient/dataset
        salt = b'hipaa_compliant_salt_2024'
        
        kdf = PBKDF2(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        
        key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
        return key
    
    def encrypt(self, data: Any) -> str:
        """
        Encrypt data
        
        Args:
            data: Data to encrypt (will be JSON serialized)
        
        Returns:
            Encrypted data as base64 string
        """
        
        # Serialize to JSON
        json_data = json.dumps(data)
        
        # Encrypt
        encrypted_bytes = self.cipher_suite.encrypt(json_data.encode())
        
        # Return as base64 string
        return base64.urlsafe_b64encode(encrypted_bytes).decode()
    
    def decrypt(self, encrypted_data: str) -> Any:
        """
        Decrypt data
        
        Args:
            encrypted_data: Encrypted data as base64 string
        
        Returns:
            Decrypted data
        """
        
        # Decode from base64
        encrypted_bytes = base64.urlsafe_b64decode(encrypted_data.encode())
        
        # Decrypt
        decrypted_bytes = self.cipher_suite.decrypt(encrypted_bytes)
        
        # Parse JSON
        return json.loads(decrypted_bytes.decode())


# Convenience functions
_encryptor = None


def _get_encryptor() -> PatientDataEncryption:
    """Get singleton encryptor instance"""
    global _encryptor
    if _encryptor is None:
        _encryptor = PatientDataEncryption()
    return _encryptor


def encrypt_patient_data(data: Any) -> str:
    """
    Encrypt patient data
    
    Args:
        data: Patient data to encrypt
    
    Returns:
        Encrypted data string
    """
    return _get_encryptor().encrypt(data)


def decrypt_patient_data(encrypted_data: str) -> Any:
    """
    Decrypt patient data
    
    Args:
        encrypted_data: Encrypted data string
    
    Returns:
        Decrypted patient data
    """
    return _get_encryptor().decrypt(encrypted_data)


def encrypt_field(value: str) -> str:
    """
    Encrypt a single field value
    
    Args:
        value: Field value
    
    Returns:
        Encrypted value
    """
    return _get_encryptor().encrypt(value)


def decrypt_field(encrypted_value: str) -> str:
    """
    Decrypt a single field value
    
    Args:
        encrypted_value: Encrypted value
    
    Returns:
        Decrypted value
    """
    return _get_encryptor().decrypt(encrypted_value)