"""
Data encryption and decryption for secure storage.
"""

from cryptography.fernet import Fernet
from typing import Dict, Any, Optional
import base64
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EncryptionManager:
    """Manage data encryption and decryption"""
    
    def __init__(self, encryption_key: Optional[bytes] = None):
        """
        Initialize encryption manager
        
        Args:
            encryption_key: Encryption key (generates new if not provided)
        """
        if encryption_key:
            self.key = encryption_key
        else:
            self.key = Fernet.generate_key()
        
        self.cipher = Fernet(self.key)
    
    def get_key(self) -> bytes:
        """Get encryption key (store securely!)"""
        return self.key
    
    def encrypt_text(self, plaintext: str) -> str:
        """
        Encrypt text
        
        Args:
            plaintext: Text to encrypt
            
        Returns:
            Encrypted text (base64 encoded)
        """
        encrypted = self.cipher.encrypt(plaintext.encode())
        return encrypted.decode()
    
    def decrypt_text(self, ciphertext: str) -> str:
        """
        Decrypt text
        
        Args:
            ciphertext: Encrypted text
            
        Returns:
            Decrypted plaintext
        """
        decrypted = self.cipher.decrypt(ciphertext.encode())
        return decrypted.decode()
    
    def encrypt_dict(self, data: Dict[str, Any]) -> str:
        """
        Encrypt dictionary
        
        Args:
            data: Dictionary to encrypt
            
        Returns:
            Encrypted data as string
        """
        import json
        json_str = json.dumps(data)
        return self.encrypt_text(json_str)
    
    def decrypt_dict(self, ciphertext: str) -> Dict[str, Any]:
        """
        Decrypt dictionary
        
        Args:
            ciphertext: Encrypted data
            
        Returns:
            Decrypted dictionary
        """
        import json
        json_str = self.decrypt_text(ciphertext)
        return json.loads(json_str)
    
    def encrypt_file(self, input_path: str, output_path: str):
        """Encrypt file"""
        with open(input_path, 'rb') as f:
            plaintext = f.read()
        
        encrypted = self.cipher.encrypt(plaintext)
        
        with open(output_path, 'wb') as f:
            f.write(encrypted)
        
        logger.info(f"Encrypted file: {input_path} -> {output_path}")
    
    def decrypt_file(self, input_path: str, output_path: str):
        """Decrypt file"""
        with open(input_path, 'rb') as f:
            ciphertext = f.read()
        
        decrypted = self.cipher.decrypt(ciphertext)
        
        with open(output_path, 'wb') as f:
            f.write(decrypted)
        
        logger.info(f"Decrypted file: {input_path} -> {output_path}")