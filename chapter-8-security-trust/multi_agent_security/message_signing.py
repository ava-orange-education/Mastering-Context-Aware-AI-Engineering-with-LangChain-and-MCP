"""
Message signing for secure agent communication.
"""

import hmac
import hashlib
from typing import Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MessageSigner:
    """Sign and verify agent messages"""
    
    def sign_message(self, message: str, secret_key: str) -> str:
        """
        Create HMAC signature for message
        
        Args:
            message: Message to sign
            secret_key: Secret signing key
            
        Returns:
            HMAC signature (hex)
        """
        signature = hmac.new(
            secret_key.encode(),
            message.encode(),
            hashlib.sha256
        ).hexdigest()
        
        return signature
    
    def verify_signature(self, message: str, signature: str, secret_key: str) -> bool:
        """
        Verify message signature
        
        Args:
            message: Original message
            signature: Claimed signature
            secret_key: Secret key
            
        Returns:
            True if signature is valid
        """
        expected_signature = self.sign_message(message, secret_key)
        
        # Use constant-time comparison
        return hmac.compare_digest(signature, expected_signature)
    
    def create_signed_message(self, message: str, sender_id: str, 
                            secret_key: str) -> Dict[str, str]:
        """
        Create message with signature
        
        Args:
            message: Message content
            sender_id: Sender identifier
            secret_key: Sender's secret key
            
        Returns:
            Signed message package
        """
        signature = self.sign_message(message, secret_key)
        
        return {
            'message': message,
            'sender_id': sender_id,
            'signature': signature
        }
    
    def verify_signed_message(self, signed_message: Dict[str, str], 
                             secret_key: str) -> Dict[str, Any]:
        """
        Verify signed message
        
        Args:
            signed_message: Message with signature
            secret_key: Expected sender's secret key
            
        Returns:
            Verification result
        """
        message = signed_message.get('message', '')
        signature = signed_message.get('signature', '')
        sender_id = signed_message.get('sender_id', '')
        
        is_valid = self.verify_signature(message, signature, secret_key)
        
        return {
            'valid': is_valid,
            'sender_id': sender_id,
            'message': message if is_valid else None
        }