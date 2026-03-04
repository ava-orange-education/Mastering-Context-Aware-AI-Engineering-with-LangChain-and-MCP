"""
Guard secure communication between agents.
"""

from cryptography.fernet import Fernet
from typing import Dict, Any
from .message_signing import MessageSigner
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CommunicationGuard:
    """Secure agent-to-agent communication"""
    
    def __init__(self):
        """Initialize communication guard"""
        self.shared_keys: Dict[str, bytes] = {}
        self.signer = MessageSigner()
    
    def establish_shared_key(self, agent_a: str, agent_b: str) -> bytes:
        """
        Establish shared encryption key between agents
        
        Args:
            agent_a: First agent
            agent_b: Second agent
            
        Returns:
            Shared key
        """
        key = Fernet.generate_key()
        
        # Store for both directions
        self.shared_keys[f"{agent_a}:{agent_b}"] = key
        self.shared_keys[f"{agent_b}:{agent_a}"] = key
        
        return key
    
    def encrypt_message(self, message: str, sender: str, recipient: str) -> str:
        """
        Encrypt message for recipient
        
        Args:
            message: Plain text message
            sender: Sender agent ID
            recipient: Recipient agent ID
            
        Returns:
            Encrypted message
        """
        key_id = f"{sender}:{recipient}"
        key = self.shared_keys.get(key_id)
        
        if not key:
            # Auto-establish key
            key = self.establish_shared_key(sender, recipient)
        
        cipher = Fernet(key)
        encrypted = cipher.encrypt(message.encode())
        
        return encrypted.decode()
    
    def decrypt_message(self, encrypted: str, sender: str, recipient: str) -> str:
        """
        Decrypt message from sender
        
        Args:
            encrypted: Encrypted message
            sender: Sender agent ID
            recipient: Recipient agent ID (self)
            
        Returns:
            Decrypted message
        """
        key_id = f"{sender}:{recipient}"
        key = self.shared_keys.get(key_id)
        
        if not key:
            raise ValueError(f"No shared key for {sender} -> {recipient}")
        
        cipher = Fernet(key)
        decrypted = cipher.decrypt(encrypted.encode())
        
        return decrypted.decode()
    
    def secure_send(self, message: str, sender: str, sender_key: str,
                   recipient: str) -> Dict[str, str]:
        """
        Send secure message (encrypted + signed)
        
        Args:
            message: Message content
            sender: Sender agent ID
            sender_key: Sender's secret key
            recipient: Recipient agent ID
            
        Returns:
            Secure message package
        """
        # Sign message
        signature = self.signer.sign_message(message, sender_key)
        
        # Encrypt message
        encrypted = self.encrypt_message(message, sender, recipient)
        
        return {
            'encrypted_message': encrypted,
            'signature': signature,
            'sender': sender,
            'recipient': recipient
        }
    
    def secure_receive(self, package: Dict[str, str], sender_key: str) -> Dict[str, Any]:
        """
        Receive and verify secure message
        
        Args:
            package: Secure message package
            sender_key: Sender's secret key for verification
            
        Returns:
            Decrypted and verified message
        """
        sender = package['sender']
        recipient = package['recipient']
        encrypted = package['encrypted_message']
        signature = package['signature']
        
        # Decrypt
        message = self.decrypt_message(encrypted, sender, recipient)
        
        # Verify signature
        is_valid = self.signer.verify_signature(message, signature, sender_key)
        
        if not is_valid:
            raise ValueError("Invalid message signature")
        
        return {
            'message': message,
            'sender': sender,
            'verified': True
        }